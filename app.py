"""
Author: Hemen Babis
Date: February 2, 2023
Description: Flask-based AI-powered To-Do List Manager. The application allows users to add, edit, delete, 
and prioritize tasks using an AI model to predict task priority based on importance, due date, and description length.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from datetime import datetime, date
import numpy as np
import joblib
from models import db, Task, CommonTask

app = Flask(__name__)

# Set up the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tasks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database with the app
db.init_app(app)

# Load the trained model
try:
    model = joblib.load('task_priority_model.pkl')
except:
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor()

def calculate_priority(task):
    try:
        task_datetime = datetime.strptime(f"{task.due_date} {task.due_time}", '%Y-%m-%d %H:%M')
        time_diff = (task_datetime - datetime.now()).total_seconds() / 3600  # Hours until due
        description_length = len(task.description)
        
        # Create feature array (importance, hours_left, description_length)
        features = np.array([[task.importance, time_diff, description_length]])
        
        # Predict priority using AI model
        priority_score = float(model.predict(features)[0])
        return min(max(priority_score, 0.0), 1.0)  # Ensure priority is between 0 and 1
    except Exception as e:
        print(f"Error calculating priority: {e}")
        return float(task.importance) / 5.0

# Create the database tables if they don't exist
with app.app_context():
    db.create_all()

@app.route('/add_task', methods=['POST'])
def add_task():
    task_id = request.form.get('task_id')
    title = request.form['title']
    description = request.form['description']
    due_date = request.form['due_date']
    due_time = request.form['due_time']
    importance = int(request.form['importance'])

    # Validate due date and time
    try:
        due_datetime = datetime.strptime(f"{due_date} {due_time}", '%Y-%m-%d %H:%M')
        if due_datetime < datetime.now():
            return jsonify({
                'success': False,
                'message': 'Due date and time cannot be in the past'
            }), 400
    except ValueError:
        return jsonify({
            'success': False,
            'message': 'Invalid date/time format'
        }), 400

    if task_id:
        task = Task.query.get(task_id)
        if task:
            task.title = title
            task.description = description
            task.due_date = due_date
            task.due_time = due_time
            task.importance = importance
            task.priority = calculate_priority(task)
            db.session.commit()
    else:
        new_task = Task(
            title=title,
            description=description,
            due_date=due_date,
            due_time=due_time,
            importance=importance
        )
        new_task.priority = calculate_priority(new_task)
        db.session.add(new_task)
        db.session.commit()

    return redirect(url_for('index'))

@app.route('/complete_task/<int:task_id>', methods=['POST'])
def complete_task(task_id):
    task = Task.query.get_or_404(task_id)
    task.completed = True
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/uncomplete_task/<int:task_id>', methods=['POST'])
def uncomplete_task(task_id):
    task = Task.query.get_or_404(task_id)
    task.completed = False
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/delete_task/<int:task_id>', methods=['POST'])
def delete_task(task_id):
    task = Task.query.get_or_404(task_id)
    db.session.delete(task)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/get_common_tasks')
def get_common_tasks():
    tasks = CommonTask.query.all()
    return jsonify([{'id': task.id, 'title': task.title} for task in tasks])

@app.route('/add_common_task', methods=['POST'])
def add_common_task():
    data = request.get_json()
    title = data.get('title')
    
    if title:
        task = CommonTask(title=title)
        db.session.add(task)
        db.session.commit()
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'message': 'Title is required'})

@app.route('/delete_common_task/<int:task_id>', methods=['POST'])
def delete_common_task(task_id):
    task = CommonTask.query.get_or_404(task_id)
    db.session.delete(task)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/')
def index():
    now = datetime.now()
    
    # Get incomplete tasks
    incomplete_tasks = Task.query.filter_by(completed=False).all()
    
    # Separate missed and current tasks
    missed_tasks = []
    current_tasks = []
    
    for task in incomplete_tasks:
        task_datetime = datetime.strptime(f"{task.due_date} {task.due_time}", '%Y-%m-%d %H:%M')
        if task_datetime < now:
            missed_tasks.append(task)
        else:
            current_tasks.append(task)
    
    # Sort tasks by importance
    missed_tasks.sort(key=lambda x: x.importance, reverse=True)
    current_tasks.sort(key=lambda x: x.importance, reverse=True)
    
    # Get completed tasks
    completed_tasks = Task.query.filter_by(completed=True).order_by(Task.importance.desc()).all()
    
    return render_template('index.html', 
                         incomplete_tasks=current_tasks, 
                         missed_tasks=missed_tasks,
                         completed_tasks=completed_tasks)

@app.route('/get_tasks')
def get_tasks():
    now = datetime.now()
    incomplete_tasks = Task.query.filter_by(completed=False).all()
    
    missed_tasks = []
    current_tasks = []
    
    for task in incomplete_tasks:
        task_datetime = datetime.strptime(f"{task.due_date} {task.due_time}", '%Y-%m-%d %H:%M')
        if task_datetime < now:
            missed_tasks.append(task)
        else:
            current_tasks.append(task)
    
    completed_tasks = Task.query.filter_by(completed=True).all()
    
    return jsonify({
        'current_tasks': [task_to_dict(t) for t in current_tasks],
        'missed_tasks': [task_to_dict(t) for t in missed_tasks],
        'completed_tasks': [task_to_dict(t) for t in completed_tasks]
    })

def task_to_dict(task):
    return {
        'id': task.id,
        'title': task.title,
        'description': task.description,
        'due_date': task.due_date,
        'due_time': task.due_time,
        'importance': task.importance,
        'completed': task.completed
    }

if __name__ == '__main__':
    app.run(debug=True)
