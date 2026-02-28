EMBEDDING_MODEL = "all-MiniLM-L6-v2"

from enum import Enum 

class MemoryTypeName(str, Enum):
    IMMEDIATE = "immediate"
    HISTORICAL = "historical"
    TEMPORAL = "temporal"
    EXPERIENTIAL = "experiential"

STALENESS_THRESHOLDS = {
    MemoryTypeName.IMMEDIATE: 7,      
    MemoryTypeName.HISTORICAL: 365,   
    MemoryTypeName.TEMPORAL: 30,       
    MemoryTypeName.EXPERIENTIAL: 180,  
}

ARCHIVE_INACTIVITY_DAYS = 90

DECAY_RATE = 0.01


RELEVANCE_WEIGHTS = {
    "semantic":   0.50,   
    "temporal":   0.25,   
    "relational": 0.25,   
}


MAX_CONTEXT_ITEMS = 10

MIN_RELEVANCE_SCORE = 0.15

EVERGREEN_IMPORTANCE_THRESHOLD = 8

EVERGREEN_TAG = "evergreen"
