#--Codigo generado con la ayuda de ChatGPT--#
import simpy
import numpy as np
import matplotlib.pyplot as plt

# 🔹 Parámetros de simulación
RANDOM_SEED = 42
INTERVALOS_LLEGADA = [10, 5, 1]  # Intervalos de llegada
CANTIDAD_PROCESOS = [25, 50, 100, 150, 200]  # Carga de trabajo
MEMORIA_RAM = 100  # RAM total disponible
INSTRUCCIONES_POR_CICLO = 3  # Instrucciones ejecutadas por unidad de tiempo
CPU_CAPACIDAD = 2  # Núcleos de CPU

np.random.seed(RANDOM_SEED)  # Fijar semilla para reproducibilidad

class Proceso:
    """Clase que representa un proceso en el sistema operativo."""
    def __init__(self, env, nombre, ram, cpu, tiempos):
        self.env = env
        self.nombre = nombre
        self.ram = ram
        self.cpu = cpu
        self.tiempos = tiempos
        self.memoria_requerida = np.random.randint(1, 11)
        self.instrucciones_restantes = np.random.randint(1, 11)
        self.tiempo_llegada = env.now
        env.process(self.ejecutar())

    def ejecutar(self):
        """Simula la ejecución del proceso en la CPU"""
        yield self.ram.get(self.memoria_requerida)  # Solicita RAM

        while self.instrucciones_restantes > 0:
            with self.cpu.request() as req:
                yield req
                instrucciones = min(INSTRUCCIONES_POR_CICLO, self.instrucciones_restantes)
                yield self.env.timeout(1)  # Simula ejecución
                self.instrucciones_restantes -= instrucciones

                # Posible espera por I/O
                if self.instrucciones_restantes > 0 and np.random.choice([True, False]):
                    yield self.env.timeout(np.random.randint(1, 6))  # Simula espera de I/O

        self.ram.put(self.memoria_requerida)  # Libera RAM al terminar
        self.tiempos.append(self.env.now - self.tiempo_llegada)

class SistemaOperativo:
    """Simulación de un sistema operativo que maneja procesos."""
    def __init__(self, env, num_procesos, intervalo, memoria_ram, capacidad_cpu):
        self.env = env
        self.ram = simpy.Container(env, init=memoria_ram, capacity=memoria_ram)
        self.cpu = simpy.Resource(env, capacity=capacidad_cpu)
        self.tiempos = []
        self.intervalo = intervalo
        self.num_procesos = num_procesos

    def generar_procesos(self):
        """Crea los procesos en el sistema según una distribución exponencial."""
        for i in range(self.num_procesos):
            Proceso(self.env, f'P-{i}', self.ram, self.cpu, self.tiempos)
            yield self.env.timeout(np.random.exponential(1.0 / self.intervalo))

    def correr_simulacion(self):
        """Ejecuta la simulación"""
        self.env.process(self.generar_procesos())
        self.env.run()
        return np.mean(self.tiempos), np.std(self.tiempos)

# 🔹 Ejecutar simulaciones y almacenar resultados
resultados = {intervalo: [] for intervalo in INTERVALOS_LLEGADA}

for intervalo in INTERVALOS_LLEGADA:
    for procesos in CANTIDAD_PROCESOS:
        env = simpy.Environment()
        so = SistemaOperativo(env, procesos, intervalo, MEMORIA_RAM, CPU_CAPACIDAD)
        prom, desv = so.correr_simulacion()
        resultados[intervalo].append((procesos, prom, desv))
        print(f'Intervalo {intervalo} | Procesos {procesos} -> Promedio: {prom:.2f}, Desviación: {desv:.2f}')

# 🔹 Graficar resultados
plt.figure(figsize=(10, 5))
for intervalo, datos in resultados.items():
    datos = np.array(datos)
    plt.plot(datos[:, 0], datos[:, 1], marker='o', label=f'Intervalo {intervalo}')

plt.xlabel('Cantidad de procesos')
plt.ylabel('Tiempo promedio en sistema')
plt.legend()
plt.title('Impacto de la carga de trabajo en el tiempo de ejecución')
plt.grid()
plt.show()