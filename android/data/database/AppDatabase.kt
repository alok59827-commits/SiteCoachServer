package com.sitecoach.data.database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.sitecoach.data.dao.DictionaryDao
import com.sitecoach.data.entity.STTCorrection
import com.sitecoach.data.entity.ValidWord

@Database(
    entities = [ValidWord::class, STTCorrection::class],
    version = 1,
    exportSchema = true
)
abstract class AppDatabase : RoomDatabase() {

    abstract fun dictionaryDao(): DictionaryDao

    companion object {
        private const val DB_NAME = "sitecoach_dictionary.db"

        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getInstance(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    DB_NAME
                )
                    .fallbackToDestructiveMigration()
                    .build()
                    .also { INSTANCE = it }
            }
        }
    }
}
