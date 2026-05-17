package com.sitecoach.data.entity

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "valid_words",
    indices = [Index(value = ["word"], unique = true)]
)
data class ValidWord(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    @ColumnInfo(name = "word")
    val word: String,

    @ColumnInfo(name = "meaning")
    val meaning: String = "",

    @ColumnInfo(name = "synonyms")
    val synonyms: String = "",

    @ColumnInfo(name = "category")
    val category: String = "General"
)
