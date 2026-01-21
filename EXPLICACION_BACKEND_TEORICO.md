# Fundamentos Teóricos de la Arquitectura Backend

## 1. Introducción

El backend del sistema FicaAsistant implementa una arquitectura de comunicación en tiempo real que integra un modelo de lenguaje de gran escala (LLM) con una interfaz conversacional web. Este documento presenta los fundamentos teóricos de cada componente arquitectónico.

---

## 2. Paradigma de Arquitectura Web: MVT (Model-View-Template)

Django implementa el patrón arquitectónico **MVT** (Model-View-Template), una variante del clásico MVC (Model-View-Controller):

| Componente | Responsabilidad | Equivalencia MVC |
|------------|-----------------|------------------|
| **Model** | Define la estructura de datos y la lógica de negocio relacionada con la persistencia. | Model |
| **View** | Contiene la lógica de procesamiento de solicitudes y la generación de respuestas. | Controller |
| **Template** | Define la presentación visual de los datos al usuario. | View |

### Justificación de la Elección
Django fue seleccionado por su madurez, documentación extensa y soporte nativo para extensiones asíncronas mediante Django Channels, lo cual es fundamental para la implementación de comunicación bidireccional en tiempo real.

---

## 3. Protocolos de Comunicación: WSGI vs ASGI

### 3.1 WSGI (Web Server Gateway Interface)

WSGI es el estándar tradicional para la comunicación entre servidores web y aplicaciones Python. Su modelo es **síncrono y sin estado**:

```
Cliente → Servidor → Aplicación WSGI → Respuesta → Cliente
         (ciclo cerrado por cada petición)
```

**Limitación**: WSGI no puede mantener conexiones persistentes, lo que impide la comunicación bidireccional continua requerida para streaming de respuestas de IA.

### 3.2 ASGI (Asynchronous Server Gateway Interface)

ASGI extiende WSGI para soportar:
- **Conexiones de larga duración**: WebSockets, HTTP/2 Server Push.
- **Concurrencia asíncrona**: Manejo de múltiples conexiones sin bloqueo mediante `async/await`.
- **Protocolos bidireccionales**: El servidor puede enviar datos al cliente sin que este lo solicite.

```
Cliente ←──────────────────────────→ Servidor ASGI
         (conexión persistente bidireccional)
```

### 3.3 Comparativa

| Característica | WSGI | ASGI |
|----------------|------|------|
| Modelo de concurrencia | Síncrono | Asíncrono |
| WebSockets | No soportado | Soportado nativamente |
| Conexiones simultáneas | Limitadas por threads | Miles (event loop) |
| Streaming de respuestas | No viable | Óptimo |

### 3.4 Diagrama Comparativo de Flujos

El siguiente diagrama ilustra visualmente la diferencia fundamental entre ambos paradigmas de comunicación:

```
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                        COMPARATIVA DE ARQUITECTURAS                                    ║
╠═══════════════════════════════════════╦═══════════════════════════════════════════════╣
║     WSGI (Tradicional - Síncrono)     ║       ASGI (Implementado - Asíncrono)         ║
╠═══════════════════════════════════════╬═══════════════════════════════════════════════╣
║                                       ║                                               ║
║  ┌─────────┐                          ║  ┌─────────┐                                  ║
║  │ CLIENTE │                          ║  │ CLIENTE │                                  ║
║  └────┬────┘                          ║  └────┬────┘                                  ║
║       │                               ║       │                                       ║
║       │ ──── Petición HTTP ────►      ║       │ ══════ Conexión WebSocket ══════      ║
║       │                               ║       │         (túnel abierto)               ║
║  ┌────▼────┐                          ║  ┌────▼────┐                                  ║
║  │SERVIDOR │  ⏳ Procesando...        ║  │SERVIDOR │◄─────────────────────────►       ║
║  │  (WSGI) │  ⏳ (bloqueado)          ║  │ (ASGI)  │  Bidireccional continuo          ║
║  └────┬────┘                          ║  └────┬────┘                                  ║
║       │                               ║       │                                       ║
║       │ ◄── Respuesta COMPLETA ──     ║       │  ◄── token ──                        ║
║       │     (toda la respuesta        ║       │  ◄── token ──                        ║
║       │      de una sola vez)         ║       │  ◄── token ──  (streaming)           ║
║       │                               ║       │  ◄── token ──                        ║
║       ▼                               ║       │  ◄── token ──                        ║
║  ╳ CONEXIÓN CERRADA                   ║       │  ◄── [done] ──                       ║
║                                       ║       │                                       ║
║  ════════════════════                 ║       │  (conexión sigue viva para           ║
║  Para nueva pregunta:                 ║       │   la siguiente pregunta)             ║
║  → Nueva conexión completa            ║       ▼                                       ║
║  → Headers repetidos                  ║  ═══════════════════════════                  ║
║  → Esperar respuesta completa         ║  Ventajas:                                    ║
║                                       ║  ✓ Sin reconexión                             ║
║                                       ║  ✓ Respuesta inmediata                        ║
║                                       ║  ✓ Menor latencia percibida                   ║
║                                       ║                                               ║
╠═══════════════════════════════════════╬═══════════════════════════════════════════════╣
║  EXPERIENCIA DE USUARIO:              ║  EXPERIENCIA DE USUARIO:                      ║
║                                       ║                                               ║
║  Usuario: "¿Qué es la matrícula?"     ║  Usuario: "¿Qué es la matrícula?"             ║
║                                       ║                                               ║
║  ⏳ Esperando... (3-5 segundos)       ║  Bot: "La"                                    ║
║  ⏳ Esperando...                      ║  Bot: "La matrícula"                          ║
║  ⏳ Esperando...                      ║  Bot: "La matrícula es"                       ║
║                                       ║  Bot: "La matrícula es el"                    ║
║  Bot: "La matrícula es el proceso     ║  Bot: "La matrícula es el proceso"            ║
║        de inscripción oficial..."     ║  Bot: "La matrícula es el proceso de..."      ║
║                                       ║                                               ║
║  (Respuesta aparece de golpe)         ║  (Respuesta aparece palabra por palabra)      ║
║                                       ║                                               ║
╚═══════════════════════════════════════╩═══════════════════════════════════════════════╝
```

**Interpretación del Diagrama**:

- **WSGI (izquierda)**: Cada interacción es un ciclo cerrado. El cliente envía una petición, el servidor procesa (bloqueando recursos), genera la respuesta completa en memoria, y finalmente la envía. La conexión se cierra inmediatamente después, requiriendo un nuevo handshake para cada mensaje.

- **ASGI (derecha)**: Se establece un túnel persistente mediante WebSocket. El servidor puede enviar fragmentos de respuesta (tokens) de forma incremental mientras el modelo de IA los genera. La conexión permanece activa, permitiendo múltiples intercambios sin sobrecarga de reconexión.

---

## 4. WebSockets: Fundamentos Teóricos

### 4.1 Definición

WebSocket es un protocolo de comunicación definido en el RFC 6455 que proporciona canales de comunicación **full-duplex** sobre una única conexión TCP. A diferencia de HTTP, donde cada interacción requiere una nueva solicitud, WebSocket mantiene una conexión persistente.

### 4.2 Handshake de Conexión

El establecimiento de una conexión WebSocket inicia como una solicitud HTTP con el header `Upgrade`:

```http
GET /ws/chat/ HTTP/1.1
Host: localhost:8000
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13
```

El servidor responde con código 101 (Switching Protocols) y la conexión se convierte en WebSocket.

#### Diagrama de Secuencia: Proceso de Handshake WebSocket

El siguiente diagrama ilustra cómo una conexión HTTP tradicional se "transforma" en un canal de comunicación en tiempo real:

```
     ┌──────────────┐                                      ┌──────────────┐
     │              │                                      │              │
     │   CLIENTE    │                                      │   SERVIDOR   │
     │  (Navegador) │                                      │   (Daphne)   │
     │              │                                      │              │
     └──────┬───────┘                                      └──────┬───────┘
            │                                                     │
            │                                                     │
    ════════╪═════════════════════════════════════════════════════╪════════
            │         FASE 1: SOLICITUD DE UPGRADE (HTTP)         │
    ════════╪═════════════════════════════════════════════════════╪════════
            │                                                     │
            │  ┌─────────────────────────────────────────────┐    │
            │  │ GET /ws/chat/ HTTP/1.1                      │    │
            │  │ Host: localhost:8000                        │    │
            │  │ Upgrade: websocket  ◄── Solicita cambio     │    │
            │  │ Connection: Upgrade                         │    │
            │  │ Sec-WebSocket-Key: dGhlIHNh...              │    │
            │  └─────────────────────────────────────────────┘    │
            │ ──────────────────────────────────────────────────► │
            │                                                     │
            │                                                     │
    ════════╪═════════════════════════════════════════════════════╪════════
            │         FASE 2: CONFIRMACIÓN DEL SERVIDOR           │
    ════════╪═════════════════════════════════════════════════════╪════════
            │                                                     │
            │    ┌─────────────────────────────────────────────┐  │
            │    │ HTTP/1.1 101 Switching Protocols            │  │
            │    │ Upgrade: websocket                          │  │
            │    │ Connection: Upgrade                         │  │
            │    │ Sec-WebSocket-Accept: s3pPLMBiTx...         │  │
            │    └─────────────────────────────────────────────┘  │
            │ ◄────────────────────────────────────────────────── │
            │                                                     │
            │  ✓ Protocolo cambiado exitosamente                  │
            │  ✓ Ya no es HTTP, ahora es WebSocket                │
            │                                                     │
            │                                                     │
    ════════╪═════════════════════════════════════════════════════╪════════
            │    FASE 3: CONEXIÓN PERSISTENTE BIDIRECCIONAL       │
    ════════╪═════════════════════════════════════════════════════╪════════
            │                                                     │
            │ ╔═══════════════════════════════════════════════════╗
            │ ║                                                   ║
            │ ║   ┌─────────────────────────────────────────┐     ║
            │ ║   │     TÚNEL WEBSOCKET ABIERTO             │     ║
            │ ║   │     (Conexión TCP persistente)          │     ║
            │ ║   └─────────────────────────────────────────┘     ║
            │ ║                                                   ║
            │ ║   ──► {"message": "Hola"}                         ║
            │ ║                                                   ║
            │ ║   ◄── {"type": "stream", "content": "¡Hola"}      ║
            │ ║   ◄── {"type": "stream", "content": "!"}          ║
            │ ║   ◄── {"type": "stream", "content": " ¿En"}       ║
            │ ║   ◄── {"type": "stream", "content": " qué"}       ║
            │ ║   ◄── {"type": "stream", "content": " puedo"}     ║
            │ ║   ◄── {"type": "stream", "content": " ayudarte?"} ║
            │ ║   ◄── {"type": "done"}                            ║
            │ ║                                                   ║
            │ ║   ──► {"message": "¿Qué es la matrícula?"}        ║
            │ ║                                                   ║
            │ ║   ◄── {"type": "stream", "content": "La"}         ║
            │ ║   ◄── {"type": "stream", "content": " matrícula"} ║
            │ ║   ◄── ...                                         ║
            │ ║                                                   ║
            │ ║   (La conexión permanece abierta indefinidamente) ║
            │ ║   (No hay overhead de reconexión entre mensajes)  ║
            │ ║                                                   ║
            │ ╚═══════════════════════════════════════════════════╝
            │                                                     │
            ▼                                                     ▼
         TIEMPO                                                TIEMPO
```

**Figura 4.1**: Diagrama de secuencia del proceso de handshake WebSocket, mostrando la transición desde una solicitud HTTP inicial hacia una conexión persistente bidireccional.

**Interpretación del Diagrama**:

1. **Fase 1 - Solicitud de Upgrade**: El cliente inicia con una petición HTTP estándar, pero incluye headers especiales (`Upgrade: websocket`) que indican su intención de cambiar de protocolo.

2. **Fase 2 - Confirmación**: El servidor valida la solicitud y responde con código **101 Switching Protocols**. Este es el último mensaje HTTP de la conexión.

3. **Fase 3 - Canal Persistente**: A partir de este momento, la conexión TCP subyacente permanece abierta. Ambas partes pueden enviar mensajes en cualquier momento sin esperar turno (full-duplex), representado por el recuadro que engloba múltiples intercambios.

### 4.3 Ventajas para Sistemas de IA Conversacional

| Aspecto | HTTP Tradicional | WebSocket |
|---------|------------------|-----------|
| Latencia percibida | Alta (polling) | Baja (push inmediato) |
| Sobrecarga de red | Headers repetidos | Mínima tras handshake |
| Experiencia de usuario | Respuesta completa | Streaming token por token |

---

## 5. Django Channels: Arquitectura de Capas

Django Channels extiende Django para manejar protocolos más allá de HTTP. Su arquitectura se compone de:

### 5.1 Capas Conceptuales

```
┌─────────────────────────────────────────────────────────┐
│                    Interface Layer                       │
│  (Daphne - Servidor ASGI)                               │
├─────────────────────────────────────────────────────────┤
│                    Routing Layer                         │
│  (ProtocolTypeRouter, URLRouter)                        │
├─────────────────────────────────────────────────────────┤
│                    Consumer Layer                        │
│  (AsyncWebsocketConsumer - Lógica de negocio)           │
├─────────────────────────────────────────────────────────┤
│                    Channel Layer                         │
│  (InMemoryChannelLayer - Comunicación entre procesos)   │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Consumers: El Patrón de Diseño

Un **Consumer** en Django Channels implementa el patrón **Handler** (manejador de eventos). Cada conexión WebSocket instancia un Consumer que:

1. **Mantiene estado**: Almacena el historial de conversación en `self.history`.
2. **Procesa eventos**: Reacciona a conexiones, mensajes y desconexiones.
3. **Es asíncrono**: Utiliza `async/await` para operaciones no bloqueantes.

```python
class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Evento: Nueva conexión establecida
        await self.accept()
        
    async def receive(self, text_data):
        # Evento: Mensaje recibido del cliente
        
    async def disconnect(self, code):
        # Evento: Conexión cerrada
```

---

## 6. Integración con Modelos de Lenguaje

### 6.1 Patrón de Carga Eagerly (Pre-carga)

El modelo de IA se carga **una única vez** al iniciar el servidor, no por cada solicitud:

```python
# Nivel de módulo (ejecutado al importar)
model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
```

**Justificación**: La carga de un modelo de 3 billones de parámetros toma 15-30 segundos y consume ~6GB de VRAM. Realizarla por cada solicitud sería inviable.

### 6.2 Generación de Texto con Streaming

La generación de respuestas utiliza el patrón **Iterator Streaming**:

```python
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
thread = Thread(target=model.generate, kwargs={"streamer": streamer})
thread.start()

for token in streamer:
    await self.send({"type": "stream", "content": token})
```

**Análisis**:
- **Problema**: `model.generate()` es una operación bloqueante de larga duración (segundos).
- **Solución**: Se ejecuta en un Thread separado mientras el Consumer itera los tokens generados.
- **Resultado**: El usuario ve la respuesta aparecer palabra por palabra, reduciendo la latencia percibida.

### 6.3 Concurrencia: Event Loop vs Threading

| Componente | Modelo de Concurrencia | Justificación |
|------------|------------------------|---------------|
| Consumer (Django) | Async/Await (Event Loop) | Manejo eficiente de I/O (WebSocket) |
| Generación IA | Threading | Operación CPU-bound intensiva |

La combinación de ambos modelos permite que la aplicación:
- Maneje cientos de conexiones WebSocket simultáneas (async).
- Ejecute generación de texto sin bloquear el event loop (thread).

---

## 7. Gestión de Estado Conversacional

### 7.1 Contexto de Conversación

El sistema mantiene un historial de mensajes en memoria para preservar el contexto:

```python
self.history = [
    {"role": "system", "content": SYSTEM_INSTRUCTION},
    {"role": "user", "content": "¿Qué son las matrículas?"},
    {"role": "assistant", "content": "Según el reglamento..."},
    # ... mensajes subsecuentes
]
```

### 7.2 Ventana de Contexto Deslizante

Para evitar exceder el límite de tokens del modelo (típicamente 4096-8192), se implementa una **ventana deslizante**:

```python
MAX_HISTORY_TURNS = 6
recent_turns = self.history[-MAX_HISTORY_TURNS:]
```

Esto garantiza que solo los últimos N turnos de conversación se envíen al modelo, manteniendo la relevancia sin agotar la memoria de contexto.

---

## 8. Diagrama de Secuencia

```
┌────────┐          ┌──────────┐          ┌──────────────┐          ┌─────────┐
│Cliente │          │ Daphne   │          │ChatConsumer  │          │ LLM     │
│(Browser│          │ (ASGI)   │          │              │          │(Llama)  │
└───┬────┘          └────┬─────┘          └──────┬───────┘          └────┬────┘
    │                    │                       │                       │
    │ WS Handshake       │                       │                       │
    │───────────────────>│                       │                       │
    │                    │ Route to Consumer     │                       │
    │                    │──────────────────────>│                       │
    │                    │                       │ connect()             │
    │<───────────────────│<──────────────────────│                       │
    │ Connection OK      │                       │                       │
    │                    │                       │                       │
    │ {"message": "..."}│                       │                       │
    │───────────────────>│──────────────────────>│                       │
    │                    │                       │ receive()             │
    │                    │                       │                       │
    │                    │                       │ Start Thread          │
    │                    │                       │──────────────────────>│
    │                    │                       │                       │ generate()
    │                    │                       │     token_1           │
    │<───────────────────│<──────────────────────│<──────────────────────│
    │ {"stream": "..."}  │                       │     token_2           │
    │<───────────────────│<──────────────────────│<──────────────────────│
    │                    │                       │     ...               │
    │                    │                       │     token_n           │
    │<───────────────────│<──────────────────────│<──────────────────────│
    │ {"done": true}     │                       │                       │
    │                    │                       │                       │
```

---

## 9. Consideraciones de Seguridad

### 9.1 Configuraciones Implementadas

| Medida | Configuración | Propósito |
|--------|---------------|-----------|
| CSRF Protection | `CSRF_TRUSTED_ORIGINS` | Previene ataques Cross-Site Request Forgery |
| Secure Cookies | `SESSION_COOKIE_SECURE = True` | Cookies solo transmitidas por HTTPS |
| SSL Proxy Header | `SECURE_PROXY_SSL_HEADER` | Detección correcta de HTTPS tras proxy |

### 9.2 Validación de Entrada

El Consumer valida los mensajes entrantes antes de procesarlos:
```python
user_msg = data.get("message", "").strip()
if not user_msg:
    return  # Ignora mensajes vacíos
```

---

## 10. Escalabilidad y Limitaciones

### 10.1 Arquitectura Actual (Monolítica)

El diseño actual es monolítico: un solo proceso ejecuta tanto el servidor web como el modelo de IA. Esto es adecuado para:
- Entornos de desarrollo y demostración.
- Despliegues de baja concurrencia (< 10 usuarios simultáneos).

### 10.2 Posibles Mejoras para Producción

| Componente | Mejora | Beneficio |
|------------|--------|-----------|
| Channel Layer | Redis en lugar de InMemory | Persistencia y escalabilidad horizontal |
| Inferencia IA | Servicio separado (vLLM, TGI) | Escalabilidad independiente |
| Balanceo de carga | Nginx + múltiples workers | Mayor throughput |

---

## 11. Conclusiones

La arquitectura backend de FicaAsistant combina tecnologías modernas para lograr:

1. **Comunicación en tiempo real** mediante WebSockets y Django Channels.
2. **Integración eficiente con IA** mediante streaming asíncrono y threading.
3. **Experiencia de usuario optimizada** con respuestas incrementales que reducen la latencia percibida.

El diseño prioriza la simplicidad y la viabilidad en hardware de consumo, mientras mantiene las puertas abiertas para futuras mejoras de escalabilidad.

---

## 12. Referencias

- Django Software Foundation. (2024). *Django Channels Documentation*. https://channels.readthedocs.io/
- Fette, I., & Melnikov, A. (2011). *The WebSocket Protocol*. RFC 6455. IETF.
- Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. Proceedings of EMNLP.
- Hugging Face. (2024). *Text Generation Inference Documentation*. https://huggingface.co/docs/text-generation-inference
