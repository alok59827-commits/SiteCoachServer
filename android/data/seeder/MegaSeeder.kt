package com.sitecoach.data.seeder

import android.content.Context
import android.util.JsonReader
import android.util.Log
import com.sitecoach.data.database.AppDatabase
import com.sitecoach.data.entity.ValidWord
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.InputStreamReader

/**
 * Memory-safe streaming seeder for the 45 MB+ master_hindi_dict.json.
 *
 * Uses [android.util.JsonReader] to stream-parse the JSON array one object
 * at a time — never loading the full file into memory — so it works on
 * devices with tight heap limits without risking OutOfMemoryError.
 *
 * Inserts are batched ([BATCH_SIZE] rows per transaction) to balance
 * write speed against memory footprint.
 *
 * Usage (from Application.onCreate or a ViewModel):
 * ```
 * lifecycleScope.launch {
 *     MegaSeeder.seedIfNeeded(applicationContext) { progress, total ->
 *         // update a progress bar (called every BATCH_SIZE inserts)
 *     }
 * }
 * ```
 */
object MegaSeeder {

    private const val TAG = "MegaSeeder"
    private const val ASSET_FILE = "master_hindi_dict.json"
    private const val BATCH_SIZE = 500
    private const val EXPECTED_TOTAL = 105_000

    /**
     * Seeds the database only when the valid_words table is empty.
     * Safe to call on every app launch — it's a no-op after the first seed.
     *
     * @param onProgress Optional callback: (insertedSoFar, estimatedTotal).
     *                   Called on [Dispatchers.IO], post to main if needed.
     */
    suspend fun seedIfNeeded(
        context: Context,
        onProgress: ((inserted: Int, estimatedTotal: Int) -> Unit)? = null
    ) {
        withContext(Dispatchers.IO) {
            val dao = AppDatabase.getInstance(context).dictionaryDao()

            val existingCount = dao.validWordCount()
            if (existingCount > 0) {
                Log.d(TAG, "Already seeded ($existingCount words). Skipping.")
                return@withContext
            }

            Log.d(TAG, "First launch — streaming $ASSET_FILE into Room DB...")
            val startMs = System.currentTimeMillis()

            var inserted = 0

            try {
                context.assets.open(ASSET_FILE).use { inputStream ->
                    JsonReader(InputStreamReader(inputStream, Charsets.UTF_8)).use { reader ->
                        inserted = streamParseAndInsert(reader, dao, onProgress)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Seeding failed at $inserted records: ${e.message}", e)
                return@withContext
            }

            val elapsedMs = System.currentTimeMillis() - startMs
            Log.d(TAG, "Seeding complete: $inserted words in ${elapsedMs}ms " +
                    "(${inserted * 1000L / maxOf(elapsedMs, 1)} words/sec)")
        }
    }

    /**
     * Stream-parse the JSON array and batch-insert into Room.
     * Returns total number of rows inserted.
     */
    private suspend fun streamParseAndInsert(
        reader: JsonReader,
        dao: com.sitecoach.data.dao.DictionaryDao,
        onProgress: ((Int, Int) -> Unit)?
    ): Int {
        val batch = ArrayList<ValidWord>(BATCH_SIZE)
        var totalInserted = 0

        reader.beginArray()

        while (reader.hasNext()) {
            kotlinx.coroutines.currentCoroutineContext().ensureActive()

            val entry = readOneEntry(reader) ?: continue
            batch.add(entry)

            if (batch.size >= BATCH_SIZE) {
                dao.insertValidWords(batch)
                totalInserted += batch.size
                batch.clear()
                onProgress?.invoke(totalInserted, EXPECTED_TOTAL)
            }
        }

        reader.endArray()

        if (batch.isNotEmpty()) {
            dao.insertValidWords(batch)
            totalInserted += batch.size
            onProgress?.invoke(totalInserted, EXPECTED_TOTAL)
        }

        return totalInserted
    }

    /**
     * Read a single JSON object `{"word":..., "meaning":..., "synonyms":..., "category":...}`
     * from the stream without buffering the whole file.
     */
    private fun readOneEntry(reader: JsonReader): ValidWord? {
        var word = ""
        var meaning = ""
        var synonyms = ""
        var category = "General"

        reader.beginObject()
        while (reader.hasNext()) {
            when (reader.nextName()) {
                "word"     -> word = reader.nextString()
                "meaning"  -> meaning = reader.nextString()
                "synonyms" -> synonyms = reader.nextString()
                "category" -> category = reader.nextString()
                else       -> reader.skipValue()
            }
        }
        reader.endObject()

        if (word.isBlank()) return null

        return ValidWord(
            word = word,
            meaning = meaning,
            synonyms = synonyms,
            category = category
        )
    }

    /**
     * Force re-seed: clears the table and re-imports from assets.
     * Useful during development or after a dictionary update.
     */
    suspend fun reseed(
        context: Context,
        onProgress: ((Int, Int) -> Unit)? = null
    ) {
        withContext(Dispatchers.IO) {
            val dao = AppDatabase.getInstance(context).dictionaryDao()
            Log.d(TAG, "Clearing valid_words table for re-seed...")
            dao.clearValidWords()
        }
        seedIfNeeded(context, onProgress)
    }
}
