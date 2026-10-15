"""infer.py

Punto de entrada para la ejecución / inferencia del problema de Machine Learning.

Responsabilidades:
- Recibir el prompt de usuario o especificación del problema (rutas de datos, objetivo, métricas, restricciones).
- Inicializar la pizarra (blackboard.md) escribiendo la especificación inicial en la sección correspondiente.
- Cargar y ejecutar el grafo del flujo de conversación definido en declarations/flows/.
- Orquestar la secuencia de agentes: dar el turno a cada agente según las aristas del grafo.
- Permitir que cada agente lea el contexto previo, ejecute sus herramientas y escriba en la pizarra.
- Guiar la ejecución hasta el agente final (consolidador), quien leerá toda la pizarra acumulada y emitirá la decisión final.
"""
