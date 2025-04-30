import os

# Obtener rutas base
root_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(root_path, '..')

# Definir rutas específicas
data_path = os.path.join(project_root, 'data')
output_path = os.path.join(project_root, 'output')
plots_path = os.path.join(project_root, 'plots')
build_output_path = os.path.join(project_root, 'build', 'output')

def get_data_file_path(filename):
    """Obtener la ruta completa para un archivo en el directorio data"""
    return os.path.join(data_path, filename)

def get_output_file_path(filename):
    """Obtener la ruta completa para un archivo en el directorio output"""
    return os.path.join(output_path, filename)

def get_plot_file_path(filename):
    """Obtener la ruta completa para un archivo en el directorio plots"""
    return os.path.join(plots_path, filename)

def get_build_output_file_path(filename):
    """Obtener la ruta completa para un archivo en el directorio build/output"""
    return os.path.join(build_output_path, filename)

def ensure_directories_exist():
    """Crear directorios necesarios si no existen"""
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(build_output_path, exist_ok=True) 