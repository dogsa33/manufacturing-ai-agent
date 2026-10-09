RAG_EVAL_CASES = [
    {
        "id": "cbm_en",
        "query": "What is condition-based maintenance?",
        "relevant": [
            {
                "source_contains": "nasa_rcmguide",
                "section_contains": "2.2 CONDITION-BASED MONITORING",
            },
            {
                "source_contains": "omguide",
                "section_contains": "5.4 Predictive Maintenance",
            },
        ],
    },
    {
        "id": "cbm_ko",
        "query": "상태 기반 정비란 무엇인가?",
        "relevant": [
            {
                "source_contains": "nasa_rcmguide",
                "section_contains": "2.2 CONDITION-BASED MONITORING",
            },
            {
                "source_contains": "omguide",
                "section_contains": "5.4 Predictive Maintenance",
            },
        ],
    },
    {
        "id": "vibration_en",
        "query": "How is vibration monitoring used for predictive maintenance?",
        "relevant": [
            {
                "source_contains": "nasa_rcmguide",
                "section_contains": "6.3 VIBRATION MONITORING",
            },
            {
                "source_contains": "omguide",
                "section_contains": "6.5.3 System Applications",
            },
        ],
    },
    {
        "id": "vibration_ko",
        "query": "진동 모니터링은 예지 정비에서 어떻게 활용되는가?",
        "relevant": [
            {
                "source_contains": "nasa_rcmguide",
                "section_contains": "6.3 VIBRATION MONITORING",
            },
            {
                "source_contains": "omguide",
                "section_contains": "6.5.3 System Applications",
            },
        ],
    },
    {
        "id": "hdf_en",
        "query": "What causes heat dissipation failure in the AI4I dataset?",
        "relevant": [
            {
                "source_contains": "AI4I",
                "section_contains": "Additional Variable Information",
            },
        ],
    },
    {
        "id": "hdf_ko",
        "query": "AI4I 데이터셋에서 열 방산 고장은 어떤 조건에서 발생하는가?",
        "relevant": [
            {
                "source_contains": "AI4I",
                "section_contains": "Additional Variable Information",
            },
        ],
    },
]