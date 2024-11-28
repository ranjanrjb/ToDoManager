from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500))
    due_date = db.Column(db.String(10), nullable=False)
    due_time = db.Column(db.String(5), nullable=False)
    importance = db.Column(db.Integer, nullable=False)
    completed = db.Column(db.Boolean, default=False)
    priority = db.Column(db.Float, default=0.0)

    def __repr__(self):
        return f"<Task {self.title}>"

class CommonTask(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"<CommonTask {self.title}>"
