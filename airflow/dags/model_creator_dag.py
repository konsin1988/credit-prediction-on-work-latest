from airflow import DAG
from airflow.operators.python import PythonOperator
from plugins.cp_model_learning import model_learning
  
from datetime import datetime

# Определение DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 12)
}

with DAG(
    'credit_prediction',
    default_args=default_args,
    catchup=False,
    max_active_runs = 1,
    description='DAG to learn model',
    schedule_interval='@daily'
) as dag:
    # Определение задач
    model_learning_op = PythonOperator(
        task_id='get_data',
        python_callable=model_learning,
        dag=dag,
    )

    # Order of tasks
    model_learning_op
