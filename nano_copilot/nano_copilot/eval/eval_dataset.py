EVAL_DATASET = [
    # TIER 1: EASY / LOCALIZED LOGIC BUGS (6 Tasks)
    {
        "task_id": "production_easy_math_zero",
        "complexity": "easy",
        "query": "Fix a ZeroDivisionError crash in compute_rolling_average inside src/core_engine/utils/math_helpers.py when empty lists are passed.",
        "test_command": "pytest tests/test_analytics.py::test_empty_metrics_rolling_average",
        "target_files": ["src/core_engine/utils/math_helpers.py"]
    },
    {
        "task_id": "production_easy_off_by_one",
        "complexity": "easy",
        "query": "Fix an off-by-one boundary truncation error inside segmented_chunk_range in src/core_engine/utils/math_helpers.py that drops the final index sequence.",
        "test_command": "pytest tests/test_analytics.py::test_chunk_range_boundaries",
        "target_files": ["src/core_engine/utils/math_helpers.py"]
    },
    {
        "task_id": "production_easy_regex_tld",
        "complexity": "easy",
        "query": "Update the email validator regex inside is_valid_secure_email in src/core_engine/utils/crypto.py to accept modern TLD strings up to 6 characters long.",
        "test_command": "pytest tests/test_auth_pipeline.py::test_modern_tld_validation",
        "target_files": ["src/core_engine/utils/crypto.py"]
    },
    {
        "task_id": "production_easy_config_env",
        "complexity": "easy",
        "query": "Modify src/core_engine/config.py to fall back to 'localhost' if the DATABASE_HOST env variable is missing, instead of crashing via an AssertionError.",
        "test_command": "pytest tests/test_architecture.py::test_system_config_fallback_logic",
        "target_files": ["src/core_engine/config.py"]
    },
    {
        "task_id": "production_easy_payload_get",
        "complexity": "easy",
        "query": "Fix the KeyError crash in extract_transaction_id inside src/core_engine/utils/payloads.py by safely reading missing dictionary keys, returning None if not present.",
        "test_command": "pytest tests/test_auth_pipeline.py::test_partial_payload_safe_extraction",
        "target_files": ["src/core_engine/utils/payloads.py"]
    },
    {
        "task_id": "production_easy_bool_cast",
        "complexity": "easy",
        "query": "Fix parse_filter_toggle in src/core_engine/utils/payloads.py. Currently, casting the string 'false' returns True, which breaks API query filters.",
        "test_command": "pytest tests/test_auth_pipeline.py::test_boolean_string_parsing_toggle",
        "target_files": ["src/core_engine/utils/payloads.py"]
    },

    # TIER 2: MEDIUM / ENCAPSULATION & RIPPLE EFFECTS (5 Tasks)
    {
        "task_id": "production_medium_exception_drift",
        "complexity": "medium",
        "query": "Define a custom exception named TokenExpiredError inside src/core_engine/exceptions.py inheriting from BaseEngineException. Then, modify verify_session_token in src/core_engine/middleware/auth.py to raise it.",
        "test_command": "pytest tests/test_auth_pipeline.py::test_domain_exception_mapping",
        "target_files": ["src/core_engine/exceptions.py", "src/core_engine/middleware/auth.py"]
    },
    {
        "task_id": "production_medium_log_context",
        "complexity": "medium",
        "query": "Fix format_structured_log in src/core_engine/middleware/telemetry.py to ensure it injects tracking attributes from the ctx_headers dictionary into the output log string.",
        "test_command": "pytest tests/test_auth_pipeline.py::test_structured_log_context_retention",
        "target_files": ["src/core_engine/middleware/telemetry.py"]
    },
    {
        "task_id": "production_medium_enum_mismatch",
        "complexity": "medium",
        "query": "Fix verify_premium_access in src/core_engine/models/user.py to correctly evaluate type equality against the UserTier Enum instead of comparing strings directly.",
        "test_command": "pytest tests/test_auth_pipeline.py::test_user_premium_enum_comparison",
        "target_files": ["src/core_engine/models/user.py"]
    },
    {
        "task_id": "production_medium_dict_mutation",
        "complexity": "medium",
        "query": "Fix the default state leak in update_request_context inside src/core_engine/config.py. It mutates the global default_headers dictionary, corrupting headers across subsequent calls.",
        "test_command": "pytest tests/test_architecture.py::test_config_shared_state_mutation",
        "target_files": ["src/core_engine/config.py"]
    },

    # TIER 3: MEDIUM-HARD / LOCAL ARCHITECTURAL SEARCH (4 Tasks)
    {
        "task_id": "production_med_hard_circular_dependency",
        "complexity": "medium_hard",
        "query": "Resolve the circular dependency import loop between src/core_engine/models/order.py and src/core_engine/services/shipping.py by abstracting contracts to src/core_engine/utils/contracts.py.",
        "test_command": "pytest tests/test_architecture.py::test_circular_imports_integrity",
        "target_files": ["src/core_engine/models/order.py", "src/core_engine/services/shipping.py", "src/core_engine/utils/contracts.py"]
    },
    {
        "task_id": "production_med_hard_thread_race",
        "complexity": "medium_hard",
        "query": "Secure acquire_cluster_slot inside src/core_engine/database.py using a threading primitive lock to prevent multi-threaded race conditions on resource allocation.",
        "test_command": "pytest tests/test_architecture.py::test_database_thread_race_conditions",
        "target_files": ["src/core_engine/database.py"]
    },
    {
        "task_id": "production_med_hard_memory_leak",
        "complexity": "medium_hard",
        "query": "Fix the memory reference leak inside unhook_metric_listener in src/core_engine/middleware/telemetry.py by removing the event keys cleanly from GLOBAL_METRIC_LISTENERS.",
        "test_command": "pytest tests/test_auth_pipeline.py::test_telemetry_listener_leak",
        "target_files": ["src/core_engine/middleware/telemetry.py"]
    },
    {
        "task_id": "production_med_hard_cache_ttl",
        "complexity": "medium_hard",
        "query": "Fix the static expiration bug inside src/core_engine/database.py. The cache baseline evaluation should check dynamic offsets relative to invocation time instead of pinning a static runtime timestamp.",
        "test_command": "pytest tests/test_architecture.py::test_runtime_cache_ttl_expiration",
        "target_files": ["src/core_engine/database.py"]
    },

    # TIER 4: HARD / PERFORMANCE & SECURITY CONTRAST (4 Tasks)
    {
        "task_id": "production_hard_memory_streaming",
        "complexity": "hard",
        "query": "Optimize generate_exported_batch inside src/core_engine/services/exporter.py to return a lazy generator expression instead of evaluating full lists in-memory.",
        "test_command": "pytest tests/test_analytics.py::test_streaming_memory_footprint",
        "target_files": ["src/core_engine/services/exporter.py"]
    },
    {
        "task_id": "production_hard_sql_injection",
        "complexity": "hard",
        "query": "Remediate SQL injection risks within execute_raw_user_query inside src/core_engine/database.py by using safe parameterized bindings instead of dynamic f-string formatting.",
        "test_command": "pytest tests/test_architecture.py::test_sql_injection_sanitization_enforcement",
        "target_files": ["src/core_engine/database.py"]
    },
    {
        "task_id": "production_hard_async_gather",
        "complexity": "hard",
        "query": "Optimize fetch_remote_metrics_batch inside src/core_engine/services/exporter.py to run concurrently using asyncio.gather instead of blocking sequentially in a linear await loop.",
        "test_command": "pytest tests/test_analytics.py::test_concurrent_metrics_batch_speed",
        "target_files": ["src/core_engine/services/exporter.py"]
    },
    {
        "task_id": "production_hard_graceful_drain",
        "complexity": "hard",
        "query": "Implement trigger_sigterm_drain inside src/core_engine/config.py to cleanly clear out elements from the active_workers_queue sequence to handle system shutdowns safely.",
        "test_command": "pytest tests/test_architecture.py::test_sigterm_graceful_queue_drain",
        "target_files": ["src/core_engine/config.py"]
    }
]