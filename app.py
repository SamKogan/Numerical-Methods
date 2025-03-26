from flask import Flask, render_template, jsonify, request, session
import numpy as np
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session

def matrix_to_latex(matrix):
    """Convert numpy matrix to LaTeX format"""
    if len(matrix.shape) == 1:
        matrix = matrix.reshape(-1, 1)
    latex = r'\begin{pmatrix} '
    for i in range(len(matrix)):
        latex += ' & '.join([f'{int(x)}' if isinstance(x, (int, np.integer)) else f'{x:.3f}' for x in matrix[i]])
        if i < len(matrix) - 1:
            latex += r' \\ '
    latex += r' \end{pmatrix}'
    return latex

def generate_random_matrix(size=3):
    """Generate a random integer matrix suitable for the numerical method"""
    matrix = np.random.randint(-5, 6, (size, size))
    for i in range(size):
        matrix[i][i] = np.sum(np.abs(matrix[i])) + np.random.randint(1, 6)
    return matrix

def make_diagonally_dominant(matrix):
    """Make a matrix diagonally dominant for Jacobi and Gauss-Seidel methods"""
    for i in range(len(matrix)):
        matrix[i][i] = np.sum(np.abs(matrix[i])) + np.random.randint(1, 6)
    return matrix

@app.route('/')
def home():
    session.clear()
    return render_template('index.html')

@app.route('/power-method')
def power_method_page():
    # Clear other methods' session data
    for key in list(session.keys()):
        if not key.startswith('power_'):
            session.pop(key)
    
    if 'power_matrix' not in session:
        matrix = generate_random_matrix()
        session['power_matrix'] = matrix.tolist()
        session['power_x'] = np.ones(3).tolist()
        session['power_eigenvalue'] = 0
        session['power_step'] = 0
        session['power_history'] = []
    
    matrix = np.array(session['power_matrix'])
    x = np.array(session['power_x'])
    eigenvalue = session['power_eigenvalue']
    history = session['power_history']
    
    return render_template('power_method.html',
                         matrix=matrix.tolist(),
                         matrix_latex=matrix_to_latex(matrix),
                         vector_latex=matrix_to_latex(x),
                         eigenvalue=eigenvalue,
                         step=session['power_step'],
                         history=history)

@app.route('/power-method/update-matrix', methods=['POST'])
def power_method_update_matrix():
    matrix = np.array(json.loads(request.form['matrix']))
    session['power_matrix'] = matrix.tolist()
    session['power_x'] = np.ones(3).tolist()
    session['power_eigenvalue'] = 0
    session['power_step'] = 0
    session['power_history'] = []
    
    return jsonify({
        'matrix': matrix.tolist(),
        'matrix_latex': matrix_to_latex(matrix),
        'vector_latex': matrix_to_latex(np.ones(3)),
        'eigenvalue': 0,
        'step': 0
    })

@app.route('/power-method/new-matrix')
def power_method_new_matrix():
    matrix = generate_random_matrix()
    session['power_matrix'] = matrix.tolist()
    session['power_x'] = np.ones(3).tolist()
    session['power_eigenvalue'] = 0
    session['power_step'] = 0
    session['power_history'] = []
    
    return jsonify({
        'matrix': matrix.tolist(),
        'matrix_latex': matrix_to_latex(matrix),
        'vector_latex': matrix_to_latex(np.ones(3)),
        'eigenvalue': 0,
        'step': 0
    })

@app.route('/power-method/step')
def power_method_step():
    matrix = np.array(session['power_matrix'])
    x = np.array(session['power_x'])
    
    # Perform one step
    x_new = matrix @ x
    eigenvalue_new = np.dot(x_new, x)
    x_new = x_new / np.linalg.norm(x_new)
    
    # Update session
    session['power_step'] += 1
    session['power_x'] = x_new.tolist()
    session['power_eigenvalue'] = float(eigenvalue_new)
    
    # Store step in history
    history_entry = {
        'vector_latex': matrix_to_latex(x_new),
        'eigenvalue': float(eigenvalue_new),
        'step': session['power_step']
    }
    session['power_history'] = session.get('power_history', []) + [history_entry]
    
    return jsonify({
        'vector_latex': matrix_to_latex(x_new),
        'eigenvalue': float(eigenvalue_new),
        'step': session['power_step']
    })

@app.route('/jacobi')
def jacobi_page():
    # Clear other methods' session data
    for key in list(session.keys()):
        if not key.startswith('jacobi_'):
            session.pop(key)
    
    if 'jacobi_matrix' not in session:
        A = generate_random_matrix()
        A = make_diagonally_dominant(A)
        b = np.random.randint(-10, 11, 3)
        session['jacobi_matrix'] = A.tolist()
        session['jacobi_b'] = b.tolist()
        session['jacobi_x'] = np.zeros(3).tolist()
        session['jacobi_step'] = 0
        session['jacobi_history'] = []
    
    A = np.array(session['jacobi_matrix'])
    b = np.array(session['jacobi_b'])
    x = np.array(session['jacobi_x'])
    history = session['jacobi_history']
    
    return render_template('jacobi.html',
                         matrix=A.tolist(),
                         b=b.tolist(),
                         matrix_latex=matrix_to_latex(A),
                         b_latex=matrix_to_latex(b),
                         x_latex=matrix_to_latex(x),
                         step=session['jacobi_step'],
                         history=history)

@app.route('/jacobi/update-matrix', methods=['POST'])
def jacobi_update_matrix():
    matrix = np.array(json.loads(request.form['matrix']))
    matrix = make_diagonally_dominant(matrix)
    b = np.array(json.loads(request.form.get('b', '[0, 0, 0]')))
    
    session['jacobi_matrix'] = matrix.tolist()
    session['jacobi_b'] = b.tolist()
    session['jacobi_x'] = np.zeros(3).tolist()
    session['jacobi_step'] = 0
    session['jacobi_history'] = []
    
    return jsonify({
        'matrix': matrix.tolist(),
        'matrix_latex': matrix_to_latex(matrix),
        'b_latex': matrix_to_latex(b),
        'x_latex': matrix_to_latex(np.zeros(3)),
        'step': 0
    })

@app.route('/jacobi/new-matrix')
def jacobi_new_matrix():
    A = generate_random_matrix()
    A = make_diagonally_dominant(A)
    b = np.random.randint(-10, 11, 3)
    session['jacobi_matrix'] = A.tolist()
    session['jacobi_b'] = b.tolist()
    session['jacobi_x'] = np.zeros(3).tolist()
    session['jacobi_step'] = 0
    session['jacobi_history'] = []
    
    return jsonify({
        'matrix': A.tolist(),
        'matrix_latex': matrix_to_latex(A),
        'b_latex': matrix_to_latex(b),
        'x_latex': matrix_to_latex(np.zeros(3)),
        'step': 0
    })

@app.route('/jacobi/step')
def jacobi_step():
    A = np.array(session['jacobi_matrix'])
    b = np.array(session['jacobi_b'])
    x = np.array(session['jacobi_x'])
    
    # Perform one Jacobi iteration
    D = np.diag(A)
    R = A - np.diagflat(D)
    x_new = (b - np.dot(R, x)) / D
    
    # Update session
    session['jacobi_step'] += 1
    session['jacobi_x'] = x_new.tolist()
    
    # Store step in history
    history_entry = {
        'x_latex': matrix_to_latex(x_new),
        'step': session['jacobi_step']
    }
    session['jacobi_history'] = session.get('jacobi_history', []) + [history_entry]
    
    return jsonify({
        'x_latex': matrix_to_latex(x_new),
        'step': session['jacobi_step']
    })

@app.route('/gauss-seidel')
def gauss_seidel_page():
    # Clear other methods' session data
    for key in list(session.keys()):
        if not key.startswith('gs_'):
            session.pop(key)
    
    if 'gs_matrix' not in session:
        A = generate_random_matrix()
        A = make_diagonally_dominant(A)
        b = np.random.randint(-10, 11, 3)
        session['gs_matrix'] = A.tolist()
        session['gs_b'] = b.tolist()
        session['gs_x'] = np.zeros(3).tolist()
        session['gs_step'] = 0
        session['gs_history'] = []
    
    A = np.array(session['gs_matrix'])
    b = np.array(session['gs_b'])
    x = np.array(session['gs_x'])
    history = session['gs_history']
    
    return render_template('gauss_seidel.html',
                         matrix=A.tolist(),
                         b=b.tolist(),
                         matrix_latex=matrix_to_latex(A),
                         b_latex=matrix_to_latex(b),
                         x_latex=matrix_to_latex(x),
                         step=session['gs_step'],
                         history=history)

@app.route('/gauss-seidel/update-matrix', methods=['POST'])
def gauss_seidel_update_matrix():
    matrix = np.array(json.loads(request.form['matrix']))
    matrix = make_diagonally_dominant(matrix)
    b = np.array(json.loads(request.form.get('b', '[0, 0, 0]')))
    
    session['gs_matrix'] = matrix.tolist()
    session['gs_b'] = b.tolist()
    session['gs_x'] = np.zeros(3).tolist()
    session['gs_step'] = 0
    session['gs_history'] = []
    
    return jsonify({
        'matrix': matrix.tolist(),
        'matrix_latex': matrix_to_latex(matrix),
        'b_latex': matrix_to_latex(b),
        'x_latex': matrix_to_latex(np.zeros(3)),
        'step': 0
    })

@app.route('/gauss-seidel/new-matrix')
def gauss_seidel_new_matrix():
    A = generate_random_matrix()
    A = make_diagonally_dominant(A)
    b = np.random.randint(-10, 11, 3)
    session['gs_matrix'] = A.tolist()
    session['gs_b'] = b.tolist()
    session['gs_x'] = np.zeros(3).tolist()
    session['gs_step'] = 0
    session['gs_history'] = []
    
    return jsonify({
        'matrix': A.tolist(),
        'matrix_latex': matrix_to_latex(A),
        'b_latex': matrix_to_latex(b),
        'x_latex': matrix_to_latex(np.zeros(3)),
        'step': 0
    })

@app.route('/gauss-seidel/step')
def gauss_seidel_step():
    A = np.array(session['gs_matrix'])
    b = np.array(session['gs_b'])
    x = np.array(session['gs_x'])
    
    # Perform one Gauss-Seidel iteration
    x_new = x.copy()
    for i in range(3):
        x_new[i] = (b[i] - np.dot(A[i,:i], x_new[:i]) 
                   - np.dot(A[i,i+1:], x[i+1:])) / A[i,i]
    
    # Update session
    session['gs_step'] += 1
    session['gs_x'] = x_new.tolist()
    
    # Store step in history
    history_entry = {
        'x_latex': matrix_to_latex(x_new),
        'step': session['gs_step']
    }
    session['gs_history'] = session.get('gs_history', []) + [history_entry]
    
    return jsonify({
        'x_latex': matrix_to_latex(x_new),
        'step': session['gs_step']
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
