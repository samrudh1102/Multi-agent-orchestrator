"""Real DAG definitions for the 7 SOTF pipeline DAGs described by
generate_airflow_mock_data.py, so they can be backfilled for real in this
local Airflow instance instead of only existing as static mock JSON.

Task bodies push the same XCom payload the mock-data generator computes for
that DAG run's data interval, by importing the generator module directly
(mounted read-only, outside the dags folder so it is never parsed as a DAG).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

MOCK_ROOT = Path("/opt/airflow/mock")


def _load_generator() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_airflow_mock_data", MOCK_ROOT / "generate_airflow_mock_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_generator()
SOTF = gen.load_sotf_dataset(MOCK_ROOT / "out")


class _Args:
    retention_days = 14


def _make_callable(dag_id: str, task_id: str):
    xcom_task_id, xcom_key = gen.XCOM_KEYS[dag_id]

    def _run(**context) -> None:
        window = gen.Window(
            sotf=SOTF,
            args=_Args(),
            start=context["data_interval_start"],
            end=context["data_interval_end"],
        )
        if task_id == xcom_task_id:
            payload = gen.PAYLOAD_BUILDERS[dag_id](window)
            context["ti"].xcom_push(key=xcom_key, value=payload)
            print(f"{dag_id}.{task_id}: pushed XCom key={xcom_key} for window {window.start}..{window.end}")
        else:
            print(f"{dag_id}.{task_id}: ran for window {window.start}..{window.end}")

    return _run


for spec in gen.DAG_SPECS:
    dag = DAG(
        dag_id=spec.dag_id,
        description=spec.description,
        schedule=spec.schedule,
        start_date=SOTF.now - __import__("datetime").timedelta(days=SOTF.days + 1),
        catchup=False,
        max_active_runs=4,
        default_args={"owner": spec.owners[0], "retries": 0},
        tags=list(spec.tags),
    )
    with dag:
        prev = None
        for task_spec in spec.tasks:
            task = PythonOperator(
                task_id=task_spec.task_id,
                python_callable=_make_callable(spec.dag_id, task_spec.task_id),
            )
            if prev is not None:
                prev >> task
            prev = task
    globals()[f"dag_{spec.dag_id}"] = dag
