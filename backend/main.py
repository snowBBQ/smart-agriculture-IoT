import asyncio
import json
import os
import shutil
import cv2
import ssl
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
import asyncpg
from gmqtt import Client as MQTTClient


from detector import GrapeDiseaseDetector

# --- Configuration from Environment Variables ---
DB_USER = os.getenv("POSTGRES_USER", "admin")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password123")
DB_NAME = os.getenv("POSTGRES_DB", "smart_vineyard")
DB_HOST = os.getenv("POSTGRES_HOST", "timescaledb")
DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))

MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "5aa9cba15b7a4cafb665e7bf650565bd.s1.eu.hivemq.cloud")
MQTT_PORT = int(os.getenv("MQTT_BROKER_PORT", 8883))
MQTT_USER = os.getenv("MQTT_USERNAME", "pythonBackend")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_USE_TLS = os.getenv("MQTT_USE_TLS", "True").lower() in ("true", "1", "yes")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "smartplant/sensors/#")

app = FastAPI(title="Smart Vineyard / Greenhouse API")

# Global connection pools and ML detector instance
db_pool = None
mqtt_client = None
detector = None

# Folder for camera images
RAW_IMAGES_DIR = "images/raw"
ANNOTATED_IMAGES_DIR = "images/annotated"
os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)


async def connect_to_db():
    global db_pool
    while db_pool is None:
        try:
            db_pool = await asyncpg.create_pool(
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                host=DB_HOST,
                port=DB_PORT
            )
            print("[DB] Connected successfully.")
            
            # Auto-create TimescaleDB hypertable if it doesn't exist
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS sensor_data (
                        time TIMESTAMPTZ NOT NULL,
                        sensor_type VARCHAR(50) NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        unit VARCHAR(20)
                    );
                """)
                # Convert to hypertable for TimescaleDB optimization
                await conn.execute("""
                    SELECT create_hypertable('sensor_data', 'time', if_not_exists => TRUE);
                """)
                print("[DB] Schema & TimescaleDB hypertable verified.")

        except Exception as e:
            print(f"[DB] Connection failed, retrying in 15s. Error: {e}")
            await asyncio.sleep(15)


async def write_to_db(sensor_type: str, value: float, unit: str):
    if db_pool:
        async with db_pool.acquire() as connection:
            await connection.execute("""
                INSERT INTO sensor_data (time, sensor_type, value, unit)
                VALUES (NOW(), $1, $2, $3)
            """, sensor_type, value, unit)
            print(f"[DB] Saved: {sensor_type} = {value} {unit}")


# --- MQTT Callbacks ---

def on_connect(client, flags, rc, properties):
    print(f"[MQTT] Connected to HiveMQ Cloud with result code: {rc}")
    client.subscribe(MQTT_TOPIC)
    print(f"[MQTT] Subscribed to topic: {MQTT_TOPIC}")


async def on_message(client, topic, payload, qos, properties):
    try:
        payload_str = payload.decode("utf-8")
        data = json.loads(payload_str)

        # Extract sensor type from topic
        sensor_type = topic.split("/")[-1]
        value = float(data.get("value"))
        unit = data.get("unit", "")

        print(f"[MQTT] Received: {sensor_type} -> {value} {unit}")
        await write_to_db(sensor_type, value, unit)

    except Exception as e:
        print(f"[MQTT] Error processing message: {e}")


# --- App Lifecycle ---

@app.on_event("startup")
async def startup_event():
    # Connect to DB
    await connect_to_db()

    # Initialize ML Detector 
    global detector
    model_file = os.getenv("ONNX_MODEL_PATH", "model/best.onnx")
    detector = GrapeDiseaseDetector(model_path=model_file)

    # Connect to HiveMQ Cloud MQTT Broker
    global mqtt_client
    mqtt_client = MQTTClient("python-backend-service")

    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    # Setup Authentication for HiveMQ Cloud
    if MQTT_USER and MQTT_PASSWORD:
        mqtt_client.set_auth_credentials(MQTT_USER, MQTT_PASSWORD)

    try:
        # SSL=True enables secure TLS connection for port 8883
        await mqtt_client.connect(MQTT_BROKER_HOST, port=MQTT_PORT, ssl=MQTT_USE_TLS)
        print(f"[MQTT] Connection initiated to {MQTT_BROKER_HOST}:{MQTT_PORT} (TLS={MQTT_USE_TLS})")
    except Exception as e:
        print(f"[MQTT] Connection failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    if db_pool:
        await db_pool.close()
    if mqtt_client:
        await mqtt_client.disconnect()


# --- HTTP Endpoints ---

@app.get("/")
def read_root():
    return {
        "status": "running", 
        "service": "Smart Greenhouse / Vineyard Backend",
        "model_loaded": detector.session is not None if detector else False
    }


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"plant_{timestamp}.jpg"
        raw_path = os.path.join(RAW_IMAGES_DIR, filename)
        annotated_path = os.path.join(ANNOTATED_IMAGES_DIR, filename)

        # Save raw received image to storage
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Save visual result with drawn bounding boxes
        detections = []
        if detector and detector.session:
            annotated_img, detections = detector.predict(raw_path, conf_threshold=0.55)
            cv2.imwrite(annotated_path, annotated_img)

        #Save findings to TimescaleDB
        if detections:
            for det in detections:
                await write_to_db(f"disease_{det['disease_name']}", det['confidence'], "conf")
        else:
            await write_to_db("disease_status", 0.0, "healthy_or_clear")

        print(f"[SERVER] Photo processed: {filename}, Detections: {len(detections)}")

        return {
            "status": "success",
            "filename": filename,
            "has_disease": len(detections) > 0,
            "detections_count": len(detections),
            "detections": detections
        }

    except Exception as e:
        print(f"[SERVER] Error during upload & inference: {str(e)}")
        return {"status": "error", "message": str(e)}
