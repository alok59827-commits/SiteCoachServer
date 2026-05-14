package com.sitecoach.data.entity

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "stt_corrections",
    indices = [Index(value = ["wrong_word"], unique = true)]
)
data class STTCorrection(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    @ColumnInfo(name = "wrong_word")
    val wrongWord: String,

    @ColumnInfo(name = "correct_word")
    val correctWord: String,

    @ColumnInfo(name = "category")
    val category: String = "General"
)
