# Deployment Guide

Production deployment instructions for the YOLOv8 Wafer Defect Detection system.

## Architecture Overview

```
                    ┌─────────────────────────────────────────────┐
                    │              Kubernetes Cluster              │
                    │                                             │
  Internet ──► ┌───┴───┐    ┌──────────┐    ┌────────────────┐   │
               │ Nginx │───►│ FastAPI  │───►│ Triton (GPU)   │   │
               │Ingress│    │ Gateway  │    │ ONNX/TensorRT  │   │
               └───┬───┘    └──────────┘    └────────────────┘   │
                   │         ┌──────────┐    ┌────────────────┐   │
                   │────────►│  React   │    │    Redis       │   │
                   │         │ Frontend │    │   (Cache)      │   │
                   │         └──────────┘    └────────────────┘   │
                   │         ┌──────────┐    ┌────────────────┐   │
                   │         │Prometheus│───►│   Grafana      │   │
                   │         └──────────┘    └────────────────┘   │
                   └──────────────────────────────────────────────┘
```

---

## Option 1: Google Cloud Platform (GCP) with GPU

Best for: running with Colab-familiar infrastructure, NVIDIA GPU access.

### Prerequisites
- GCP account with billing enabled
- `gcloud` CLI installed
- Docker installed locally

### Step 1: Create a GPU VM

```bash
# Create a VM with NVIDIA T4 GPU
gcloud compute instances create wafer-detection-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE \
    --metadata=startup-script='#!/bin/bash
        # Install NVIDIA drivers
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update
        sudo apt-get install -y nvidia-driver-535 nvidia-container-toolkit docker.io docker-compose-plugin
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
    '

# Open firewall ports
gcloud compute firewall-rules create wafer-detection-ports \
    --allow tcp:3000,tcp:3001,tcp:8080,tcp:9090 \
    --target-tags=wafer-detection

gcloud compute instances add-tags wafer-detection-gpu \
    --zone=us-central1-a --tags=wafer-detection
```

### Step 2: Deploy the stack

```bash
# SSH into the VM
gcloud compute ssh wafer-detection-gpu --zone=us-central1-a

# Clone the repo
git clone https://github.com/Rajendar-Muddasani-2/yolo-object-detection.git
cd yolo-object-detection

# Set production environment variables
export API_KEY=$(openssl rand -hex 32)
export JWT_SECRET=$(openssl rand -hex 32)
echo "API_KEY=$API_KEY" > .env
echo "JWT_SECRET=$JWT_SECRET" >> .env

# Pull Git LFS files (ONNX model)
git lfs pull

# Start the full stack
docker compose up -d

# Verify
docker compose ps
curl http://localhost:8080/health
```

### Step 3: Access the services

```bash
# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe wafer-detection-gpu \
    --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "Frontend:   http://$EXTERNAL_IP:3000"
echo "API:        http://$EXTERNAL_IP:8080/docs"
echo "Grafana:    http://$EXTERNAL_IP:3001  (admin/admin)"
echo "Prometheus: http://$EXTERNAL_IP:9090"
```

### Cost Estimate (GCP)
| Resource | Type | Cost/hr |
|----------|------|---------|
| VM | n1-standard-4 + T4 | ~$0.95/hr |
| Boot disk | 100 GB SSD | ~$17/month |
| **Spot VM** | Same (preemptible) | **~$0.35/hr** |

> Use spot/preemptible VMs for testing: add `--provisioning-model=SPOT` to the create command.

---

## Option 2: Kubernetes (GKE) Deployment

Best for: production-scale, auto-scaling, multi-replica serving.

### Prerequisites
- GCP account with GKE API enabled
- `kubectl` and `gcloud` CLI

### Step 1: Create GKE cluster with GPU node pool

```bash
# Create cluster
gcloud container clusters create wafer-detection-cluster \
    --zone=us-central1-a \
    --num-nodes=2 \
    --machine-type=e2-standard-4

# Add GPU node pool
gcloud container node-pools create gpu-pool \
    --cluster=wafer-detection-cluster \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --num-nodes=1 \
    --min-nodes=0 \
    --max-nodes=3 \
    --enable-autoscaling

# Install NVIDIA GPU drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
```

### Step 2: Push images to Artifact Registry

```bash
# Create registry
gcloud artifacts repositories create wafer-detection \
    --repository-format=docker --location=us-central1

# Build and push
PROJECT_ID=$(gcloud config get project)
REGISTRY=us-central1-docker.pkg.dev/$PROJECT_ID/wafer-detection

docker build -t $REGISTRY/api:latest .
docker build -t $REGISTRY/frontend:latest ./frontend
docker push $REGISTRY/api:latest
docker push $REGISTRY/frontend:latest
```

### Step 3: Apply Kubernetes manifests

```bash
kubectl apply -f k8s/
```

See [k8s/](../k8s/) directory for all manifests.

### Step 4: Verify deployment

```bash
kubectl get pods -n wafer-detection
kubectl get svc -n wafer-detection

# Get external IP of the LoadBalancer
kubectl get svc frontend -n wafer-detection -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

---

## Option 3: Docker Compose on Cloud VM (Simplest)

For a quick cloud deployment without K8s complexity.

### AWS EC2 with GPU

```bash
# Launch p3.2xlarge (V100) or g4dn.xlarge (T4)
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --instance-type g4dn.xlarge \
    --key-name your-key \
    --security-group-ids sg-xxxx \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]'

# SSH in, install Docker + NVIDIA toolkit, clone repo, docker compose up
```

### Azure VM with GPU

```bash
az vm create \
    --resource-group wafer-detection-rg \
    --name wafer-detection-vm \
    --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts:latest \
    --size Standard_NC4as_T4_v3 \
    --admin-username azureuser \
    --generate-ssh-keys

# SSH in, install Docker + NVIDIA toolkit, clone repo, docker compose up
```

---

## Kubernetes Manifests

Create the `k8s/` directory with these files:

### k8s/namespace.yaml
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: wafer-detection
```

### k8s/triton-deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton
  namespace: wafer-detection
spec:
  replicas: 1
  selector:
    matchLabels:
      app: triton
  template:
    metadata:
      labels:
        app: triton
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:24.01-py3
        command: ["tritonserver", "--model-repository=/models", "--strict-model-config=false"]
        ports:
        - containerPort: 8000
        - containerPort: 8001
        - containerPort: 8002
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
        volumeMounts:
        - name: model-store
          mountPath: /models
        readinessProbe:
          httpGet:
            path: /v2/health/ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-store
        persistentVolumeClaim:
          claimName: model-store-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: triton
  namespace: wafer-detection
spec:
  selector:
    app: triton
  ports:
  - name: http
    port: 8000
  - name: grpc
    port: 8001
  - name: metrics
    port: 8002
```

### k8s/api-deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: wafer-detection
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: REGISTRY/api:latest  # Replace with your registry
        ports:
        - containerPort: 8080
        env:
        - name: TRITON_URL
          value: "triton:8000"
        - name: AUTH_ENABLED
          value: "true"
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: api-key
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: jwt-secret
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
          requests:
            memory: "512Mi"
            cpu: "500m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 15
---
apiVersion: v1
kind: Service
metadata:
  name: api
  namespace: wafer-detection
spec:
  selector:
    app: api
  ports:
  - port: 8080
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: wafer-detection
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### k8s/frontend-deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: wafer-detection
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: REGISTRY/frontend:latest  # Replace with your registry
        ports:
        - containerPort: 80
        resources:
          limits:
            memory: "256Mi"
            cpu: "250m"
---
apiVersion: v1
kind: Service
metadata:
  name: frontend
  namespace: wafer-detection
spec:
  type: LoadBalancer
  selector:
    app: frontend
  ports:
  - port: 80
    targetPort: 80
```

### k8s/secrets.yaml
```yaml
# Generate with: kubectl create secret generic api-secrets --from-literal=api-key=$(openssl rand -hex 32) --from-literal=jwt-secret=$(openssl rand -hex 32) -n wafer-detection
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
  namespace: wafer-detection
type: Opaque
data:
  api-key: ""      # base64 encoded
  jwt-secret: ""   # base64 encoded
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_ENABLED` | `false` | Enable JWT/API key authentication |
| `API_KEY` | `dev-api-key-change-in-prod` | API key for authentication |
| `JWT_SECRET` | `dev-jwt-secret-change-in-prod` | JWT signing secret |
| `JWT_EXPIRE_MINUTES` | `60` | JWT token lifetime |
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window (seconds) |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins (comma-separated) |
| `TRITON_URL` | `localhost:8000` | Triton server address |

---

## Monitoring in Production

1. **Grafana** auto-loads the wafer defect dashboard at startup
2. **Prometheus** scrapes Triton (`:8002/metrics`) and FastAPI (`:8080/metrics`)
3. Set up alerts in Grafana for:
   - Inference latency P95 > 50ms
   - Error rate > 1%
   - GPU utilization > 90% for 5 min (scale up trigger)
   - GPU memory > 85%

---

## Load Testing in Production

```bash
# Install locust
pip install locust

# Run against cloud deployment
locust -f tests/load_test.py --host https://your-api-endpoint.com \
    --headless -u 100 -r 20 --run-time 300s --csv results/load_test

# Generate HTML report
locust -f tests/load_test.py --host https://your-api-endpoint.com \
    --headless -u 50 -r 10 --run-time 60s --html results/load_test_report.html
```
