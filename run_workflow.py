"""
Complete Workflow Runner
Execute all tasks sequentially with proper error handling and logging
"""

import os
import sys
import time
import subprocess
from datetime import datetime

class WorkflowRunner:
    """Run the complete face mask detection workflow"""
    
    def __init__(self):
        self.start_time = None
        self.results = {}
        
    def log(self, message, level="INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def run_task(self, task_name, script_path, description):
        """
        Run a single task
        
        Args:
            task_name: Name of the task
            script_path: Path to the Python script
            description: Task description
        """
        self.log("=" * 60)
        self.log(f"TASK: {task_name}")
        self.log(f"Description: {description}")
        self.log("=" * 60)
        
        start = time.time()
        
        try:
            # Check if script exists
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Script not found: {script_path}")
            
            # Run script
            self.log(f"Executing: python {script_path}")
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Log output
            if result.stdout:
                print(result.stdout)
            
            duration = time.time() - start
            self.log(f"✓ Task completed in {duration:.2f} seconds", "SUCCESS")
            
            self.results[task_name] = {
                'status': 'SUCCESS',
                'duration': duration
            }
            
            return True
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - start
            self.log(f"✗ Task failed after {duration:.2f} seconds", "ERROR")
            self.log(f"Error: {e.stderr}", "ERROR")
            
            self.results[task_name] = {
                'status': 'FAILED',
                'duration': duration,
                'error': str(e)
            }
            
            return False
            
        except Exception as e:
            duration = time.time() - start
            self.log(f"✗ Unexpected error: {str(e)}", "ERROR")
            
            self.results[task_name] = {
                'status': 'ERROR',
                'duration': duration,
                'error': str(e)
            }
            
            return False
    
    def run_all(self, skip_training=False):
        """
        Run all tasks sequentially
        
        Args:
            skip_training: Skip time-consuming training tasks
        """
        self.start_time = time.time()
        
        self.log("=" * 60)
        self.log("FACE MASK DETECTION - COMPLETE WORKFLOW")
        self.log("=" * 60)
        self.log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Skip Training: {skip_training}")
        self.log("=" * 60)
        
        # Define tasks
        tasks = [
            {
                'name': 'Task 1: Problem Definition',
                'script': 'task1_problem_definition/analyze_dataset.py',
                'description': 'Analyze dataset and define problem',
                'required': True
            },
            {
                'name': 'Task 2: Data Preprocessing',
                'script': 'task2_data_preprocessing/preprocess_data.py',
                'description': 'Preprocess images and apply augmentation',
                'required': True
            },
            {
                'name': 'Task 3: Model Training',
                'script': 'task3_model_training/train_model.py',
                'description': 'Train custom CNN model',
                'required': not skip_training
            },
            {
                'name': 'Task 4: Model Evaluation',
                'script': 'task4_evaluation/evaluate_model.py',
                'description': 'Evaluate model performance',
                'required': not skip_training
            },
            {
                'name': 'Task 6: Transfer Learning',
                'script': 'task6_advanced_optimization/transfer_learning.py',
                'description': 'Train MobileNetV2 and compare models',
                'required': not skip_training
            }
        ]
        
        # Run tasks
        for task in tasks:
            if not task['required']:
                self.log(f"\nSkipping {task['name']} (training disabled)")
                continue
            
            success = self.run_task(
                task['name'],
                task['script'],
                task['description']
            )
            
            if not success:
                self.log(f"\n⚠ Task failed: {task['name']}", "WARNING")
                user_input = input("Continue with remaining tasks? (y/n): ")
                if user_input.lower() != 'y':
                    self.log("Workflow stopped by user", "INFO")
                    break
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print workflow summary"""
        total_duration = time.time() - self.start_time
        
        self.log("\n" + "=" * 60)
        self.log("WORKFLOW SUMMARY")
        self.log("=" * 60)
        
        success_count = sum(1 for r in self.results.values() if r['status'] == 'SUCCESS')
        failed_count = sum(1 for r in self.results.values() if r['status'] == 'FAILED')
        error_count = sum(1 for r in self.results.values() if r['status'] == 'ERROR')
        
        for task_name, result in self.results.items():
            status = result['status']
            duration = result['duration']
            
            status_symbol = "✓" if status == "SUCCESS" else "✗"
            self.log(f"{status_symbol} {task_name}: {status} ({duration:.2f}s)")
        
        self.log("\n" + "=" * 60)
        self.log(f"Total Tasks: {len(self.results)}")
        self.log(f"Successful: {success_count}")
        self.log(f"Failed: {failed_count}")
        self.log(f"Errors: {error_count}")
        self.log(f"Total Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        self.log("=" * 60)
        
        if success_count == len(self.results):
            self.log("\n✓ ALL TASKS COMPLETED SUCCESSFULLY!", "SUCCESS")
            self.log("\nNext steps:")
            self.log("1. Run Flask app: python task5_frontend/app.py")
            self.log("2. Run FastAPI: python task7_deployment/api.py")
            self.log("3. Deploy using docker-compose")
        else:
            self.log("\n⚠ SOME TASKS FAILED", "WARNING")
            self.log("Review the errors above and fix before proceeding")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run the complete Face Mask Detection workflow"
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip time-consuming training tasks (Task 3, 4, 6)'
    )
    parser.add_argument(
        '--task',
        type=str,
        help='Run specific task only (e.g., task1, task2, etc.)'
    )
    
    args = parser.parse_args()
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    runner = WorkflowRunner()
    
    if args.task:
        # Run specific task
        task_map = {
            'task1': ('Task 1', 'task1_problem_definition/analyze_dataset.py', 'Dataset analysis'),
            'task2': ('Task 2', 'task2_data_preprocessing/preprocess_data.py', 'Data preprocessing'),
            'task3': ('Task 3', 'task3_model_training/train_model.py', 'Model training'),
            'task4': ('Task 4', 'task4_evaluation/evaluate_model.py', 'Model evaluation'),
            'task6': ('Task 6', 'task6_advanced_optimization/transfer_learning.py', 'Transfer learning'),
        }
        
        if args.task in task_map:
            name, script, desc = task_map[args.task]
            runner.run_task(name, script, desc)
        else:
            runner.log(f"Unknown task: {args.task}", "ERROR")
            runner.log("Available tasks: task1, task2, task3, task4, task6")
    else:
        # Run all tasks
        runner.run_all(skip_training=args.skip_training)

if __name__ == "__main__":
    main()
