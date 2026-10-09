EVAL_CASES = [

    # =====================================================
    # 1. Basic Process Summary
    # =====================================================

    {
        "id": "summary_l",
        "description": "Product Type L 기본 통계 조회",

        "question": (
            "Product Type L의 전체 샘플 수와 "
            "고장 건수, 고장률을 알려줘."
        ),

        "required_tools": [
            "process_summary",
        ],

        "forbidden_tools": [],

        "answer_patterns": [
            r"6,?000",
            r"235",
            r"3\.92\s*%",
        ],

        "max_tool_calls": 1,
    },


    # =====================================================
    # 2. Normal vs Failure Comparison
    # =====================================================

    {
        "id": "torque_comparison",
        "description": "Torque 정상/고장 비교",

        "question": (
            "Torque가 정상 제품과 고장 제품에서 "
            "어떻게 다른지 비교해서 설명해줘."
        ),

        "required_tools": [
            "compare_process_condition",
        ],

        "forbidden_tools": [],

        "answer_patterns": [
            r"39\.63",
            r"50\.17",
            r"10\.54",
        ],

        # compare_process_condition 하나면 충분
        "max_tool_calls": 1,
    },


    # =====================================================
    # 3. Failure Prediction
    # =====================================================

    {
        "id": "failure_prediction",
        "description": "특정 제조조건 고장 예측",

        "question": (
            "다음 제조조건의 고장 위험을 분석해줘. "
            "Product Type은 L, "
            "Air temperature는 301.0, "
            "Process temperature는 310.5, "
            "Rotational speed는 1300, "
            "Torque는 65.0, "
            "Tool wear는 200이야."
        ),

        "required_tools": [
            "predict_machine_failure",
        ],

        "forbidden_tools": [],

        "answer_patterns": [
            r"80\.33\s*%",
        ],

        # 설명 Tool까지 자율적으로 사용하는 것은 허용
        "max_tool_calls": 2,
    },


    # =====================================================
    # 4. Prediction + Explanation
    # =====================================================

    {
        "id": "prediction_explanation",
        "description": "고장 예측 및 위험요인 분석",

        "question": (
            "다음 제조조건의 고장 위험을 먼저 예측하고, "
            "왜 위험하게 판단됐는지도 주요 변수 기준으로 "
            "분석해줘. "
            "Product Type은 L, "
            "Air temperature는 301.0, "
            "Process temperature는 310.5, "
            "Rotational speed는 1300, "
            "Torque는 65.0, "
            "Tool wear는 200이야."
        ),

        "required_tools": [
            "predict_machine_failure",
            "explain_machine_failure",
        ],

        "forbidden_tools": [],

        "answer_patterns": [
            r"80\.33\s*%",
            r"75\.66",
            r"51(?:\.0+)?",
            r"29(?:\.0+)?",
        ],

        "max_tool_calls": 2,
    },


    # =====================================================
    # 5. Missing Inputs
    # =====================================================

    {
        "id": "missing_prediction_inputs",
        "description": "예측 입력값 부족 시 추측 방지",

        "question": (
            "Product Type L이고 Torque가 65인데 "
            "고장 위험이 높은지 예측해줘."
        ),

        # 값이 부족하므로 prediction Tool을
        # 호출하면 안 됨
        "required_tools": [],

        "forbidden_tools": [
            "predict_machine_failure",
            "explain_machine_failure",
        ],

        "answer_patterns": [],

        "max_tool_calls": 0,
    },


    # =====================================================
    # 6. Visualization
    # =====================================================

    {
        "id": "torque_chart",
        "description": "Torque 정상/고장 분포 시각화",

        "question": (
            "Torque의 정상 제품과 고장 제품 분포를 "
            "그래프로 만들어줘."
        ),

        "required_tools": [
            "feature_distribution_chart",
        ],

        "forbidden_tools": [],

        "answer_patterns": [],

        "max_tool_calls": 1,
    },
]