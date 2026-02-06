# Kubernetes Deployment Guide - Customer Churn Prediction API

This guide provides step-by-step instructions for deploying a FastAPI-based ML model service on Kubernetes using Kind (Kubernetes in Docker).

---

## Table of Contents
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Deployment Files](#deployment-files)
- [Deployment Steps](#deployment-steps)
- [Accessing the Application](#accessing-the-application)
- [Useful Commands](#useful-commands)
- [Troubleshooting](#troubleshooting)
- [Cleanup](#cleanup)

---

## Environment Setup

### Prerequisites
- Docker installed and running
- Command-line access (Terminal/PowerShell/WSL)

### Install kubectl

kubectl is the Kubernetes command-line tool.

**Windows (with Chocolatey):**
```bash
choco install kubernetes-cli
```

**macOS:**
```bash
brew install kubectl
```

**Linux(Recommended):**
```bash
cd 
mkdir -p bin && cd bin
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
cd
export PATH="${PATH}:${HOME}/bin"
echo 'export PATH="${PATH}:${HOME}/bin"' >> ~/.bashrc
```

**Verify installation:**
```bash
kubectl version --client
```

### Install Kind

Kind (Kubernetes in Docker) allows you to run Kubernetes clusters locally.

**Windows (with Chocolatey):**
```bash
choco install kind
```

**macOS:**
```bash
brew install kind
```

**Linux:**
```bash
curl -Lo ${HOME}/bin/kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ${HOME}/bin/kind
```

**Verify installation:**
```bash
kind version
```

---

## Project Structure

```
k8s/
├── kind-config.yaml       # Kind cluster configuration
├── models-pv-pvc.yaml    # PersistentVolume and PersistentVolumeClaim
├── deployment.yaml        # Application deployment
├── service.yaml          # Service to expose the application
└── deploy.sh            # Automated deployment script
```

---

## Deployment Files

### 1. `k8s/kind-config.yaml`

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraMounts:
  - hostPath: /mnt/d/Churn_Customer/NeuronetiX-Hackathon-Project/models
    containerPath: /data/models
```

⚠️ **Important:** Update `hostPath` to match your actual models directory path.

### 2. `k8s/models-pv-pvc.yaml`

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: models-pv
spec:
  storageClassName: manual
  capacity:
    storage: 1Gi
  accessModes:
    - ReadOnlyMany
  hostPath:
    path: "/data/models"
    type: Directory
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
spec:
  storageClassName: manual
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 1Gi
```

### 3. `k8s/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-churn
  labels:
    app: fastapi-churn
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fastapi-churn
  template:
    metadata:
      labels:
        app: fastapi-churn
    spec:
      containers:
        - name: fastapi-churn
          image: omarsabri12/churn-fastapi:latest
          ports:
            - containerPort: 5000
          imagePullPolicy: Always
          volumeMounts:
            - name: models-storage
              mountPath: /app/models
              readOnly: true
      volumes:
        - name: models-storage
          persistentVolumeClaim:
            claimName: models-pvc
```

### 4. `k8s/service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: fastapi-churn-service
  labels:
    app: fastapi-churn
spec:
  type: NodePort
  selector:
    app: fastapi-churn
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
      nodePort: 30080
```

---

## Deployment Steps

### Step 1: Navigate to k8s Directory

```bash
cd /path/to/your/project/k8s
```

### Step 2: Create Kind Cluster

```bash
kind create cluster --name churn --config kind-config.yaml
```

**Verify cluster:**
```bash
kubectl cluster-info --context kind-churn
kubectl get nodes
```

### Step 3: Apply PersistentVolume and PersistentVolumeClaim

```bash
kubectl apply -f models-pv-pvc.yaml
```

**Verify:**
```bash
kubectl get pv
kubectl get pvc
```

Both should show `STATUS: Bound`

### Step 4: Deploy the Application

```bash
kubectl apply -f deployment.yaml
```

**Wait for pods to be ready:**
```bash
kubectl wait --for=condition=ready pod -l app=fastapi-churn --timeout=120s
```

**Check status:**
```bash
kubectl get pods
```

### Step 5: Create Service

```bash
kubectl apply -f service.yaml
```

**Verify:**
```bash
kubectl get service
```

---

## Accessing the Application

### Method 1: Port Forwarding (Recommended)

```bash
kubectl port-forward service/fastapi-churn-service 8080:80
```

Keep this terminal open. Then access:
- **Swagger UI:** http://localhost:8080/docs
- **API endpoint:** http://localhost:8080

### Method 2: Test with curl

```bash
# In another terminal
curl http://localhost:8080/health
```

---

## Useful Commands

### View Resources

```bash
# Get all pods
kubectl get pods

# Get all services
kubectl get svc

# Get all resources
kubectl get all
```

### View Logs

```bash
# View logs from all pods
kubectl logs -l app=fastapi-churn --tail=50

# Follow logs in real-time
kubectl logs -l app=fastapi-churn -f

# Logs from specific pod
kubectl logs <pod-name>
```

### Describe Resources

```bash
# Pod details
kubectl describe pod <pod-name>

# Deployment details
kubectl describe deployment fastapi-churn

# PVC details
kubectl describe pvc models-pvc
```

### Execute Commands in Pod

```bash
# Get shell access
kubectl exec -it <pod-name> -- /bin/sh

# Check if models are mounted
kubectl exec -it <pod-name> -- ls -la /app/models/without_smoteenn/

# List files
kubectl exec -it <pod-name> -- ls -la /app/models/
```

### Scale Deployment

```bash
# Scale to 3 replicas
kubectl scale deployment fastapi-churn --replicas=3

# Scale to 1 replica
kubectl scale deployment fastapi-churn --replicas=1
```

### Update Deployment

```bash
# Apply changes from file
kubectl apply -f deployment.yaml

# Restart deployment (rolling restart)
kubectl rollout restart deployment fastapi-churn

# Check rollout status
kubectl rollout status deployment fastapi-churn
```

---

## Troubleshooting

### Issue 1: Pods in CrashLoopBackOff

```bash
# Check logs
kubectl logs <pod-name>

# Check pod events
kubectl describe pod <pod-name>
```

**Common causes:**
- Models not found in `/app/models/`
- Application startup error
- Missing dependencies

### Issue 2: Models Not Found

```bash
# Check if models are mounted in pod
kubectl exec -it <pod-name> -- ls -la /app/models/without_smoteenn/

# Check if extraMounts worked in Kind
docker exec churn-control-plane ls -la /data/models/without_smoteenn/

# Verify on your local machine
ls -la /mnt/d/Churn_Customer/NeuronetiX-Hackathon-Project/models/without_smoteenn/
```

**Solution:** 
- Ensure `hostPath` in `kind-config.yaml` is correct
- Recreate cluster with correct configuration

### Issue 3: PVC Status Pending

```bash
kubectl describe pvc models-pvc
```

**Common causes:**
- PV not created
- StorageClass mismatch
- Access mode incompatibility

**Solution:**
```bash
# Delete and recreate
kubectl delete -f models-pv-pvc.yaml
kubectl apply -f models-pv-pvc.yaml
```

### Issue 4: Cannot Access Application

```bash
# Check if pods are running
kubectl get pods

# Check service
kubectl get svc

# Check logs
kubectl logs -l app=fastapi-churn
```

**Solution:** Use port-forward command

### View Events

```bash
# See recent events
kubectl get events --sort-by='.lastTimestamp'
```

---

## Cleanup

### Delete Specific Resources

```bash
# Delete in reverse order
kubectl delete -f service.yaml
kubectl delete -f deployment.yaml
kubectl delete -f models-pv-pvc.yaml
```

### Delete the Entire Cluster

```bash
kind delete cluster --name churn
```

### Verify Deletion

```bash
# Check if cluster is deleted
kind get clusters
```

---

## Automated Deployment Script

### Create `k8s/deploy.sh`

```bash
#!/bin/bash
set -e

echo "🚀 Starting Kubernetes deployment..."

# Delete old cluster
echo "🗑️  Deleting old cluster..."
kind delete cluster --name churn 2>/dev/null || true

# Create new cluster
echo "🔨 Creating new cluster..."
kind create cluster --name churn --config kind-config.yaml

# Wait for cluster to be ready
echo "⏳ Waiting for cluster..."
kubectl wait --for=condition=Ready nodes --all --timeout=60s

# Apply PV and PVC
echo "💾 Applying PV and PVC..."
kubectl apply -f models-pv-pvc.yaml

# Wait for PVC to be bound
echo "⏳ Waiting for PVC to bind..."
kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/models-pvc --timeout=30s

# Apply Deployment
echo "🚢 Deploying application..."
kubectl apply -f deployment.yaml

# Apply Service
echo "🌐 Creating service..."
kubectl apply -f service.yaml

# Wait for pods
echo "⏳ Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=fastapi-churn --timeout=120s

# Show status
echo "✅ Deployment complete!"
echo ""
echo "📊 Current status:"
kubectl get pods
echo ""
kubectl get svc

echo ""
echo "📝 Useful commands:"
echo "  View logs:    kubectl logs -l app=fastapi-churn -f"
echo "  Get shell:    kubectl exec -it \$(kubectl get pod -l app=fastapi-churn -o jsonpath='{.items[0].metadata.name}') -- /bin/sh"
echo "  Port forward: kubectl port-forward service/fastapi-churn-service 8080:80"
echo ""
echo "🌐 Access the application:"
echo "  Run: kubectl port-forward service/fastapi-churn-service 8080:80"
echo "  Then open: http://localhost:8080/docs"
```

### Make it Executable

```bash
chmod +x deploy.sh
```

### Run the Script

```bash
./deploy.sh
```

---

## Kubernetes Concepts

### What is Kind?
- Runs Kubernetes clusters locally using Docker containers
- Perfect for testing and development
- Fast cluster creation/deletion

### What is a Pod?
- Smallest deployable unit in Kubernetes
- Contains one or more containers
- Shares network and storage

### What is a Deployment?
- Manages a set of identical pods
- Ensures desired number of replicas are running
- Handles rolling updates and rollbacks

### What is a Service?
- Exposes pods to network traffic
- Provides stable endpoint (IP/DNS)
- Load balances across pods

### What is a PersistentVolume?
- Storage resource in the cluster
- Independent of pod lifecycle
- Can be mounted by pods

### What is a PersistentVolumeClaim?
- Request for storage by a pod
- Binds to a PersistentVolume
- Abstracts storage details from pods

---

## Best Practices

1. **Always use resource limits** to prevent resource exhaustion
2. **Use readiness and liveness probes** for health checking
3. **Version your Docker images** instead of using `latest`
4. **Use ConfigMaps and Secrets** for configuration
5. **Monitor logs regularly**
6. **Test locally with Kind** before production deployment

---

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Kind Documentation](https://kind.sigs.k8s.io/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## Support

If you encounter issues:
1. Check the logs: `kubectl logs -l app=fastapi-churn`
2. Describe the pod: `kubectl describe pod <pod-name>`
3. Check events: `kubectl get events --sort-by='.lastTimestamp'`
4. Verify models are mounted: `kubectl exec -it <pod-name> -- ls -la /app/models/`

---

**Happy Deploying! 🚀**