package com.sitecoach.data.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.sitecoach.data.entity.STTCorrection
import com.sitecoach.data.entity.ValidWord

@Dao
interface DictionaryDao {

    // ── ValidWord inserts ──

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertValidWord(word: ValidWord): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertValidWords(words: List<ValidWord>)

    // ── STTCorrection inserts ──

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertCorrection(correction: STTCorrection): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertCorrections(corrections: List<STTCorrection>)

    // ── Batch lookups (zero-latency via IN clause) ──

    /**
     * Returns corrections for words that match known wrong-words.
     * Room compiles this into a single parameterised SQL statement,
     * so it executes in one round-trip regardless of list size.
     *
     * SQLite has a default variable limit of 999; the caller must
     * chunk larger lists (see [STTProcessor]).
     */
    @Query("SELECT * FROM stt_corrections WHERE wrong_word IN (:words)")
    suspend fun findCorrections(words: List<String>): List<STTCorrection>

    /**
     * Validates words against the dictionary in a single batch query.
     */
    @Query("SELECT * FROM valid_words WHERE word IN (:words)")
    suspend fun findValidWords(words: List<String>): List<ValidWord>

    // ── Single-word lookups ──

    @Query("SELECT * FROM stt_corrections WHERE wrong_word = :word LIMIT 1")
    suspend fun findCorrectionForWord(word: String): STTCorrection?

    @Query("SELECT * FROM valid_words WHERE word = :word LIMIT 1")
    suspend fun findValidWord(word: String): ValidWord?

    // ── Counts (useful for seeder first-launch check) ──

    @Query("SELECT COUNT(*) FROM valid_words")
    suspend fun validWordCount(): Int

    @Query("SELECT COUNT(*) FROM stt_corrections")
    suspend fun correctionCount(): Int

    // ── Bulk delete (for re-seeding) ──

    @Query("DELETE FROM valid_words")
    suspend fun clearValidWords()

    @Query("DELETE FROM stt_corrections")
    suspend fun clearCorrections()
}
