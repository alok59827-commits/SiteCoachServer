package com.sitecoach.data.processor

import com.sitecoach.data.dao.DictionaryDao
import com.sitecoach.data.entity.ValidWord
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Zero-Latency Layer 1 STT Correction & Vocabulary System.
 *
 * Processing pipeline (single pass, batched DB queries):
 *   Priority 1 → Replace known wrong words via [stt_corrections] table
 *   Priority 2 → Validate remaining words against [valid_words] table
 *   Priority 3 → Collect unknown words + meanings of valid words
 *
 * Usage:
 * ```
 * val processor = STTProcessor(database.dictionaryDao())
 * val result = processor.process("raw STT transcript here")
 * // result.correctedTranscript  → cleaned string
 * // result.validWords           → list of ValidWord with meanings
 * // result.unknownWords         → words not in either table
 * ```
 */
class STTProcessor(private val dao: DictionaryDao) {

    companion object {
        /**
         * SQLite binds at most 999 variables per statement.
         * We chunk word lists to stay within this limit.
         */
        private const val SQLITE_VAR_LIMIT = 900
    }

    /**
     * Immutable result of processing an STT transcript.
     */
    data class ProcessedResult(
        val correctedTranscript: String,
        val validWords: List<ValidWord>,
        val unknownWords: List<String>,
        val correctionsMade: Map<String, String>
    )

    /**
     * Main entry point. Accepts a raw STT transcript string,
     * performs all three priority passes, and returns the result.
     */
    suspend fun process(rawTranscript: String): ProcessedResult =
        withContext(Dispatchers.IO) {
            val originalTokens = tokenize(rawTranscript)
            if (originalTokens.isEmpty()) {
                return@withContext ProcessedResult(
                    correctedTranscript = rawTranscript,
                    validWords = emptyList(),
                    unknownWords = emptyList(),
                    correctionsMade = emptyMap()
                )
            }

            val uniqueWords = originalTokens.map { it.lowercase() }.distinct()

            // ── Priority 1: Batch-lookup corrections ──
            val correctionMap = batchFindCorrections(uniqueWords)

            // Apply corrections to the token list
            val correctedTokens = originalTokens.map { token ->
                correctionMap[token.lowercase()]?.let { correction ->
                    preserveCase(token, correction)
                } ?: token
            }

            // Determine which words still need validation
            val correctedUniqueWords = correctedTokens
                .map { it.lowercase() }
                .distinct()

            // ── Priority 2: Batch-validate against dictionary ──
            val validWordMap = batchFindValidWords(correctedUniqueWords)

            // ── Priority 3: Separate unknown words ──
            val validWordEntries = mutableListOf<ValidWord>()
            val unknownWords = mutableListOf<String>()
            val seen = mutableSetOf<String>()

            for (token in correctedTokens) {
                val key = token.lowercase()
                if (key in seen) continue
                seen.add(key)

                val validEntry = validWordMap[key]
                if (validEntry != null) {
                    validWordEntries.add(validEntry)
                } else {
                    unknownWords.add(token)
                }
            }

            ProcessedResult(
                correctedTranscript = correctedTokens.joinToString(" "),
                validWords = validWordEntries,
                unknownWords = unknownWords,
                correctionsMade = correctionMap
            )
        }

    // ── Private helpers ──

    private fun tokenize(text: String): List<String> {
        return text
            .split(Regex("[\\s]+"))
            .map { it.trim() }
            .filter { it.isNotBlank() }
    }

    /**
     * Batch-query corrections, chunking to respect SQLite variable limits.
     * Returns a map of wrongWord(lowercase) → correctWord.
     */
    private suspend fun batchFindCorrections(
        words: List<String>
    ): Map<String, String> {
        val result = mutableMapOf<String, String>()
        words.chunked(SQLITE_VAR_LIMIT).forEach { chunk ->
            dao.findCorrections(chunk).forEach { correction ->
                result[correction.wrongWord.lowercase()] = correction.correctWord
            }
        }
        return result
    }

    /**
     * Batch-query valid words, chunking to respect SQLite variable limits.
     * Returns a map of word(lowercase) → ValidWord entity.
     */
    private suspend fun batchFindValidWords(
        words: List<String>
    ): Map<String, ValidWord> {
        val result = mutableMapOf<String, ValidWord>()
        words.chunked(SQLITE_VAR_LIMIT).forEach { chunk ->
            dao.findValidWords(chunk).forEach { validWord ->
                result[validWord.word.lowercase()] = validWord
            }
        }
        return result
    }

    /**
     * Attempt to preserve the original casing pattern when applying
     * a correction (e.g., if original was uppercase, keep replacement uppercase).
     */
    private fun preserveCase(original: String, replacement: String): String {
        return when {
            original.all { it.isUpperCase() } -> replacement.uppercase()
            original.firstOrNull()?.isUpperCase() == true ->
                replacement.replaceFirstChar { it.uppercase() }
            else -> replacement
        }
    }
}
