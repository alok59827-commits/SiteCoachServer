package com.sitecoach.data.seeder

import android.content.Context
import android.util.JsonReader
import android.util.Log
import com.sitecoach.data.database.AppDatabase
import com.sitecoach.data.entity.STTCorrection
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.InputStreamReader

/**
 * Streaming seeder for stt_corrections_seed.json.
 *
 * Same OOM-safe approach as MegaSeeder.kt: uses [android.util.JsonReader]
 * to parse the array entry-by-entry and batches inserts of 500 rows per
 * transaction. Safe to call on every app launch — no-op once seeded.
 *
 * Usage:
 * ```
 * lifecycleScope.launch {
 *     STTCorrectionSeeder.seedIfNeeded(applicationContext) { done, total ->
 *         // update a progress bar
 *     }
 * }
 * ```
 *
 * JSON schema this seeder expects (matches the Python builder):
 * ```
 * [
 *   {
 *     "wrongWord":   "जगह रहा",
 *     "correctWord": "अजगरहा",
 *     "category":    "PROPER_NOUN",
 *     "confidence":  1.0,
 *     "source":      "expert_xlsx"  // optional
 *   }
 * ]
 * ```
 */
object STTCorrectionSeeder {

    private const val TAG = "STTCorrectionSeeder"
    private const val ASSET_FILE = "stt_corrections_seed.json"
    private const val BATCH_SIZE = 500
    private const val EXPECTED_TOTAL = 300

    /**
     * Minimum confidence we accept from the seed file. Anything below
     * this is treated as too noisy to apply on-device.
     */
    private const val MIN_CONFIDENCE = 0.6

    suspend fun seedIfNeeded(
        context: Context,
        onProgress: ((inserted: Int, estimatedTotal: Int) -> Unit)? = null
    ) {
        withContext(Dispatchers.IO) {
            val dao = AppDatabase.getInstance(context).dictionaryDao()

            val existingCount = dao.correctionCount()
            if (existingCount > 0) {
                Log.d(TAG, "Already seeded ($existingCount corrections). Skipping.")
                return@withContext
            }

            Log.d(TAG, "First launch — streaming $ASSET_FILE into Room DB...")
            val startMs = System.currentTimeMillis()
            var inserted = 0

            try {
                context.assets.open(ASSET_FILE).use { inputStream ->
                    JsonReader(InputStreamReader(inputStream, Charsets.UTF_8))
                        .use { reader ->
                            inserted = streamParseAndInsert(reader, dao, onProgress)
                        }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Seeding failed at $inserted records: ${e.message}", e)
                return@withContext
            }

            val elapsedMs = System.currentTimeMillis() - startMs
            val rate = inserted * 1000L / maxOf(elapsedMs, 1)
            Log.d(TAG, "Seeding complete: $inserted corrections in ${elapsedMs}ms ($rate/s)")
        }
    }

    private suspend fun streamParseAndInsert(
        reader: JsonReader,
        dao: com.sitecoach.data.dao.DictionaryDao,
        onProgress: ((Int, Int) -> Unit)?
    ): Int {
        val batch = ArrayList<STTCorrection>(BATCH_SIZE)
        var totalInserted = 0

        reader.beginArray()

        while (reader.hasNext()) {
            kotlinx.coroutines.currentCoroutineContext().ensureActive()

            val entry = readOneEntry(reader) ?: continue
            batch.add(entry)

            if (batch.size >= BATCH_SIZE) {
                dao.insertCorrections(batch)
                totalInserted += batch.size
                batch.clear()
                onProgress?.invoke(totalInserted, EXPECTED_TOTAL)
            }
        }

        reader.endArray()

        if (batch.isNotEmpty()) {
            dao.insertCorrections(batch)
            totalInserted += batch.size
            onProgress?.invoke(totalInserted, EXPECTED_TOTAL)
        }

        return totalInserted
    }

    private fun readOneEntry(reader: JsonReader): STTCorrection? {
        var wrong = ""
        var correct = ""
        var category = "General"
        var confidence = 1.0

        reader.beginObject()
        while (reader.hasNext()) {
            when (reader.nextName()) {
                "wrongWord"   -> wrong = reader.nextString()
                "correctWord" -> correct = reader.nextString()
                "category"    -> category = reader.nextString()
                "confidence"  -> confidence = reader.nextDouble()
                else          -> reader.skipValue()
            }
        }
        reader.endObject()

        if (wrong.isBlank() || correct.isBlank()) return null
        if (confidence < MIN_CONFIDENCE) return null

        return STTCorrection(
            wrongWord = wrong,
            correctWord = correct,
            category = category
        )
    }

    /** Force re-seed: clears the table and re-imports from assets. */
    suspend fun reseed(
        context: Context,
        onProgress: ((Int, Int) -> Unit)? = null
    ) {
        withContext(Dispatchers.IO) {
            val dao = AppDatabase.getInstance(context).dictionaryDao()
            Log.d(TAG, "Clearing stt_corrections table for re-seed...")
            dao.clearCorrections()
        }
        seedIfNeeded(context, onProgress)
    }
}
