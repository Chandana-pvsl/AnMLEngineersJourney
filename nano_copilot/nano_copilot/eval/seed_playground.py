import os
import shutil
from pathlib import Path

def setup_expanded_playground():
    base_dir = Path("./eval_playground")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(exist_ok=True)
    
    files_to_create = {
        # =====================================================================
        # CONFIGURATION & LIFECYCLE LAYER
        # =====================================================================
        "pytest.ini": (
            "[pytest]\n"
            "pythonpath = src\n"
            "testpaths = tests\n"
        ),
        
        "src/core_engine/config.py": (
            "import os\n\n"
            "class Settings:\n"
            "    PROJECT_NAME: str = 'CoreEngine'\n"
            "    DATABASE_HOST: str = os.environ.get('DATABASE_HOST')\n"
            "    # BUG (Easy_06): Throws error instead of falling back cleanly\n"
            "    if not DATABASE_HOST:\n"
            "        raise AssertionError('DATABASE_HOST missing')\n\n"
            "    def __init__(self):\n"
            "        # BUG (Med_09): Shared state object dict mutation vulnerability\n"
            "        self.default_headers = {'Content-Type': 'application/json'}\n"
            "        self.active_workers_queue = [1, 2, 3]\n\n"
            "    def update_request_context(self, trace_id: str):\n"
            "        headers = self.default_headers\n"
            "        headers['X-Trace-ID'] = trace_id\n"
            "        return headers\n\n"
            "    def trigger_sigterm_drain(self):\n"
            "        # BUG (Hard_06): Drops records abruptly without clearing active_workers_queue\n"
            "        pass\n\n"
            "settings = Settings()\n"
        ),
        
        "src/core_engine/exceptions.py": (
            "class BaseEngineException(Exception): pass\n"
            "class InvalidContextError(BaseEngineException): pass\n"
        ),
        
        "src/core_engine/database.py": (
            "import time\n"
            "import datetime\n\n"
            "class ThreadLocalMemoryDatabase:\n"
            "    def __init__(self):\n"
            "        self.allocated_resources = set()\n"
            "        # BUG (Med_Hard_07): Calculated once at instantiation, locking the TTL value forever\n"
            "        self.cache_ttl_baseline = time.time() + 3600\n\n"
            "    def acquire_cluster_slot(self, resource_id: int) -> bool:\n"
            "        # BUG (Med_Hard_05): Missing thread-lock synchronization gate\n"
            "        if resource_id in self.allocated_resources: return False\n"
            "        time.sleep(0.02)\n"
            "        self.allocated_resources.add(resource_id)\n"
            "        return True\n\n"
            "    def is_cache_expired(self) -> bool:\n"
            "        return time.time() > self.cache_ttl_baseline\n\n"
            "    def execute_raw_user_query(self, input_string: str) -> str:\n"
            "        # BUG (Hard_05): Critical Raw SQL Injection format vulnerability\n"
            "        return f\"SELECT * FROM system_logs WHERE user = '{input_string}';\"\n\n"
            "db_engine = ThreadLocalMemoryDatabase()\n"
        ),

        # =====================================================================
        # MODELS LAYER
        # =====================================================================
        "src/core_engine/models/base.py": (
            "from dataclasses import dataclass, field\n"
            "import datetime\n\n"
            "@dataclass\n"
            "class BaseDomainModel:\n"
            "    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())\n"
        ),
        
        "src/core_engine/models/user.py": (
            "from dataclasses import dataclass\n"
            "from enum import Enum\n"
            "from core_engine.models.base import BaseDomainModel\n\n"
            "class UserTier(Enum):\n"
            "    FREE = 'free'\n"
            "    PREMIUM = 'premium'\n\n"
            "@dataclass\n"
            "class UserProfile(BaseDomainModel):\n"
            "    user_id: int\n"
            "    username: str\n"
            "    tier: UserTier = UserTier.FREE\n\n"
            "    def verify_premium_access(self, evaluation_string: str) -> bool:\n"
            "        # BUG (Med_08): Compares Enum object directly to raw String object, always evaluating False\n"
            "        return self.tier == evaluation_string\n"
        ),

        # =====================================================================
        # MIDDLEWARE & PROCESSING UTILITIES LAYER
        # =====================================================================
        "src/core_engine/middleware/auth.py": (
            "def verify_session_token(is_expired: bool) -> bool:\n"
            "    if is_expired: raise ValueError('Session has expired')\n"
            "    return True\n"
        ),
        
        "src/core_engine/middleware/telemetry.py": (
            "GLOBAL_METRIC_LISTENERS = {}\n\n"
            "def hook_metric_listener(event_name: str, callback_fn):\n"
            "    GLOBAL_METRIC_LISTENERS[event_name] = callback_fn\n\n"
            "def unhook_metric_listener(event_name: str):\n"
            "    pass\n\n"
            "def format_structured_log(message: str, ctx_headers: dict) -> str:\n"
            "    # BUG (Med_07): System drops logging headers during formatting string execution\n"
            "    return f\"[LOG] {message}\"\n"
        ),
        
        "src/core_engine/utils/payloads.py": (
            "def extract_transaction_id(payload: dict) -> str:\n"
            "    # BUG (Easy_07): Direct key lookup raises a KeyError if keys are partial or missing\n"
            "    return payload['metadata']['transaction_id']\n\n"
            "def parse_filter_toggle(value_string: str) -> bool:\n"
            "    # BUG (Easy_08): Explicitly casting 'false' evaluates to True in Python\n"
            "    return bool(value_string)\n"
        ),
        
        "src/core_engine/utils/crypto.py": (
            "import re\n"
            "def is_valid_secure_email(email: str) -> bool:\n"
            "    pattern = r'^[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,4}$'\n"
            "    return bool(re.match(pattern, email.lower()))\n"
        ),
        
        "src/core_engine/utils/math_helpers.py": (
            "def compute_rolling_average(numbers: list) -> float:\n"
            "    return sum(numbers) / len(numbers)\n\n"
            "def segmented_chunk_range(data: list, step: int) -> list:\n"
            "    chunks = []\n"
            "    for i in range(0, len(data) - 1, step):\n"
            "        chunks.append(data[i:i + step])\n"
            "    return chunks\n"
        ),

        # =====================================================================
        # HEAVY BUSINESS SERVICES LAYER
        # =====================================================================
        "src/core_engine/services/billing.py": (
            "def process_transaction_charge(amount: float) -> float:\n"
            "    return amount + 2.50\n\n"
            "class IdempotentPaymentConsumer:\n"
            "    def __init__(self):\n"
            "        # BUG (Hard_07): Ephemeral memory array vanishes if process cycles or errors clear class instance\n"
            "        self.processed_ids = []\n\n"
            "    def process_idempotent_charge(self, event_id: str, amount: float) -> bool:\n"
            "        if event_id in self.processed_ids: return False\n"
            "        self.processed_ids.append(event_id)\n"
            "        return True\n"
        ),
        
        "src/core_engine/services/exporter.py": (
            "import asyncio\n"
            "import types\n\n"
            "class DataStreamExporter:\n"
            "    def __init__(self, data_source):\n"
            "        self.source = data_source\n\n"
            "    def generate_exported_batch(self) -> list:\n"
            "        return [item * 2 for item in self.source]\n\n"
            "    async def fetch_remote_metrics_batch(self, targets: list) -> list:\n"
            "        results = []\n"
            "        for target in targets:\n"
            "            # BUG (Med_Hard_08): Executes sequentially using blocking await loop iterations\n"
            "            await asyncio.sleep(0.01)\n"
            "            results.append(f\"data:{target}\")\n"
            "        return results\n"
        ),

        "src/core_engine/models/order.py": (
            "from core_engine.services.shipping import calculate_delivery_cost\n\n"
            "class ClientOrder:\n"
            "    def __init__(self, weight: float): self.weight = weight\n"
            "    def get_shipping_rate(self): return calculate_delivery_cost(self.weight)\n"
        ),
        
        "src/core_engine/services/shipping.py": (
            "from core_engine.models.order import ClientOrder\n\n"
            "def calculate_delivery_cost(weight: float) -> float:\n"
            "    return weight * 1.5\n"
        ),
        
        "src/core_engine/utils/contracts.py": "# Shared interfaces file boundary\n",

        # =====================================================================
        # OVERHAULED TEST FIXTURES MATRIX
        # =====================================================================
        "tests/test_analytics.py": (
            "import pytest\n"
            "import types\n"
            "import asyncio\n"
            "from core_engine.utils.math_helpers import compute_rolling_average, segmented_chunk_range\n"
            "from core_engine.services.exporter import DataStreamExporter\n\n"
            "def test_empty_metrics_rolling_average():\n"
            "    assert compute_rolling_average([]) == 0.0\n\n"
            "def test_chunk_range_boundaries():\n"
            "    assert len(segmented_chunk_range([10, 20, 30, 40, 50, 60], 2)) == 3\n\n"
            "def test_streaming_memory_footprint():\n"
            "    exporter = DataStreamExporter(range(100))\n"
            "    assert isinstance(exporter.generate_exported_batch(), types.GeneratorType)\n\n"
            "@pytest.mark.asyncio\n"
            "async def test_concurrent_metrics_batch_speed():\n"
            "    exporter = DataStreamExporter([])\n"
            "    start = asyncio.get_event_loop().time()\n"
            "    await exporter.fetch_remote_metrics_batch(list(range(20)))\n"
            "    duration = asyncio.get_event_loop().time() - start\n"
            "    # If tasks run concurrently via gather, it must complete in under 0.05 seconds\n"
            "    assert duration < 0.05\n"
        ),
        
        "tests/test_auth_pipeline.py": (
            "import pytest\n"
            "from core_engine.utils.crypto import is_valid_secure_email\n"
            "from core_engine.middleware.auth import verify_session_token\n"
            "from core_engine.middleware.telemetry import hook_metric_listener, unhook_metric_listener, format_structured_log, GLOBAL_METRIC_LISTENERS\n"
            "from core_engine.utils.payloads import extract_transaction_id, parse_filter_toggle\n"
            "from core_engine.models.user import UserProfile, UserTier\n\n"
            "def test_modern_tld_validation():\n"
            "    assert is_valid_secure_email('infra@cloud.engine') is True\n\n"
            "def test_domain_exception_mapping():\n"
            "    from core_engine.exceptions import TokenExpiredError\n"
            "    with pytest.raises(TokenExpiredError): verify_session_token(is_expired=True)\n\n"
            "def test_telemetry_listener_leak():\n"
            "    hook_metric_listener('pulse', lambda: None)\n"
            "    unhook_metric_listener('pulse')\n"
            "    assert 'pulse' not in GLOBAL_METRIC_LISTENERS\n\n"
            "def test_partial_payload_safe_extraction():\n"
            "    assert extract_transaction_id({'id': 1}) is None\n\n"
            "def test_boolean_string_parsing_toggle():\n"
            "    assert parse_filter_toggle('false') is False\n\n"
            "def test_structured_log_context_retention():\n"
            "    res = format_structured_log('Engine up', {'X-Trace-ID': '42'})\n"
            "    assert '42' in res\n\n"
            "def test_user_premium_enum_comparison():\n"
            "    user = UserProfile(user_id=1, username='vip', tier=UserTier.PREMIUM)\n"
            "    assert user.verify_premium_access('premium') is True\n"
        ),
        
        "tests/test_architecture.py": (
            "import pytest\n"
            "import os\n"
            "import time\n"
            "import threading\n"
            "from core_engine.database import db_engine\n"
            "from core_engine.config import settings\n"
            "from core_engine.services.billing import IdempotentPaymentConsumer\n\n"
            "def test_system_config_fallback_logic():\n"
            "    if 'DATABASE_HOST' in os.environ: del os.environ['DATABASE_HOST']\n"
            "    assert settings.DATABASE_HOST == 'localhost'\n\n"
            "def test_circular_imports_integrity():\n"
            "    from core_engine.models.order import ClientOrder\n"
            "    assert True\n\n"
            "def test_database_thread_race_conditions():\n"
            "    res = []\n"
            "    def t():\n"
            "        if db_engine.acquire_cluster_slot(99): res.append(True)\n"
            "    threads = [threading.Thread(target=t) for _ in range(5)]\n"
            "    for x in threads: x.start()\n"
            "    for x in threads: x.join()\n"
            "    assert len(res) == 1\n\n"
            "def test_config_shared_state_mutation():\n"
            "    c1 = settings.update_request_context('abc')\n"
            "    c2 = settings.update_request_context('xyz')\n"
            "    assert 'abc' not in c2\n\n"
            "def test_runtime_cache_ttl_expiration():\n"
            "    assert db_engine.is_cache_expired() is False\n"
            "    time.sleep(0.02)\n"
            "    # Mock time advancement into the future\n"
            "    db_engine.cache_ttl_baseline -= 4000\n"
            "    assert db_engine.is_cache_expired() is True\n\n"
            "def test_sql_injection_sanitization_enforcement():\n"
            "    sql = db_engine.execute_raw_user_query(\"admin' OR '1'='1\")\n"
            "    assert \"'admin'\" not in sql and (\"?\" in sql or \"%s\" in sql or \"kwargs\" in sql)\n\n"
            "def test_sigterm_graceful_queue_drain():\n"
            "    settings.trigger_sigterm_drain()\n"
            "    assert len(settings.active_workers_queue) == 0\n\n"
            "def test_payment_idempotency_persistence():\n"
            "    consumer = IdempotentPaymentConsumer()\n"
            "    assert consumer.process_idempotent_charge('evt_1', 10.0) is True\n"
            "    # Re-instantiating a clean tracker simulation across persistent ledger checks\n"
            "    del consumer\n"
            "    from core_engine.services.billing import IdempotentPaymentConsumer\n"
            "    new_consumer = IdempotentPaymentConsumer()\n"
            "    assert new_consumer.process_idempotent_charge('evt_1', 10.0) is False\n"
        )
    }

    for rel_path, content in files_to_create.items():
        path = base_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    os.system(f"cd {base_dir} && git init && git add . && git commit -m 'feat: complete production v2 database'")
    print("🚀 Real-Life Production Playground Overhauled and seeded successfully with 19 targets!")

if __name__ == "__main__":
    setup_expanded_playground()