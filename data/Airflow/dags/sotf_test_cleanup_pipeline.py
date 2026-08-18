from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

def detect_orphaned_pods():
    print("Scanning Kubernetes cluster for OOMKilled or orphaned SOTF runner pods...")

def terminate_runner_pod():
    print("Terminating pod SOTF-Runner-Pod-4 due to high memory usage (>82%)...")

def reset_socket_registry():
    print("Resetting socket connection pool for SOTF orchestrator master.")

default_args = {
    "owner": "sotf_team",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="sotf_test_cleanup_pipeline",
    default_args=default_args,
    description="Clean up dangling k8s worker pods and reset socket connections upon SOTF test failures.",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["sotf", "cleanup", "k8s"],
) as dag:

    t1 = PythonOperator(task_id="detect_orphaned_pods", python_callable=detect_orphaned_pods)
    t2 = PythonOperator(task_id="terminate_runner_pod", python_callable=terminate_runner_pod)
    t3 = PythonOperator(task_id="reset_socket_registry", python_callable=reset_socket_registry)

    t1 >> t2 >> t3