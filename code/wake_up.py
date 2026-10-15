"""wake_up.py

Módulo de inicialización ("despertar") del sistema y registro de agentes.

Responsabilidades:
- Cargar las variables de entorno desde .env (claves de API, timeouts, configuraciones).
- Escanear y validar las declaraciones en declarations/ (agentes, tools, flujos de conversación).
- Emparejar cada herramienta declarada con su respectiva función ejecutable en code/tools/.
- Inicializar las instancias de los agentes asignándoles su rol, modelo, herramientas y permisos sobre la pizarra.
- Dejar todo el entorno listo y en espera para que infer.py comience la ejecución.
"""
