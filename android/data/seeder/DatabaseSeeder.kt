package com.sitecoach.data.seeder

import android.content.Context
import android.util.Log
import com.sitecoach.data.database.AppDatabase
import com.sitecoach.data.entity.ValidWord
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray

/**
 * Reads `valid_words.json` from the Android `assets/` folder on the
 * first app launch and bulk-inserts all records into the [ValidWord] table.
 *
 * Usage (from Application.onCreate or a ViewModel):
 * ```
 * lifecycleScope.launch {
 *     DatabaseSeeder.seedIfNeeded(applicationContext)
 * }
 * ```
 */
object DatabaseSeeder {

    private const val TAG = "DatabaseSeeder"
    private const val ASSET_FILE = "valid_words.json"
    private const val BATCH_SIZE = 500

    /**
     * Seeds the database only when the valid_words table is empty.
     * Safe to call on every app launch — it's a no-op after the first seed.
     */
    suspend fun seedIfNeeded(context: Context) {
        withContext(Dispatchers.IO) {
            val dao = AppDatabase.getInstance(context).dictionaryDao()

            if (dao.validWordCount() > 0) {
                Log.d(TAG, "Database already seeded (${dao.validWordCount()} words). Skipping.")
                return@withContext
            }

            Log.d(TAG, "First launch detected. Seeding database from $ASSET_FILE...")
            val startTime = System.currentTimeMillis()

            try {
                val jsonString = context.assets
                    .open(ASSET_FILE)
                    .bufferedReader()
                    .use { it.readText() }

                val jsonArray = JSONArray(jsonString)
                val totalEntries = jsonArray.length()
                Log.d(TAG, "Parsed $totalEntries entries from JSON.")

                val batch = mutableListOf<ValidWord>()

                for (i in 0 until totalEntries) {
                    val obj = jsonArray.getJSONObject(i)
                    val word = obj.optString("word", "").trim()
                    if (word.isEmpty()) continue

                    batch.add(
                        ValidWord(
                            word = word,
                            meaning = obj.optString("meaning", ""),
                            synonyms = obj.optString("synonyms", ""),
                            category = obj.optString("category", "General")
                        )
                    )

                    if (batch.size >= BATCH_SIZE) {
                        dao.insertValidWords(batch)
                        batch.clear()
                    }
                }

                if (batch.isNotEmpty()) {
                    dao.insertValidWords(batch)
                }

                val elapsed = System.currentTimeMillis() - startTime
                Log.d(TAG, "Seeding complete: ${dao.validWordCount()} words in ${elapsed}ms.")
            } catch (e: Exception) {
                Log.e(TAG, "Seeding failed: ${e.message}", e)
            }
        }
    }

    /**
     * Force re-seed: clears the table and re-imports from assets.
     * Useful during development or after a dictionary update.
     */
    suspend fun reseed(context: Context) {
        withContext(Dispatchers.IO) {
            val dao = AppDatabase.getInstance(context).dictionaryDao()
            dao.clearValidWords()
            Log.d(TAG, "Cleared valid_words table. Re-seeding...")
        }
        seedIfNeeded(context)
    }
}
