# Kubernetes

Manifests in `k8s/` deploy the orchestrator and Python sidecar as separate workloads. Redis is assumed to be provided externally (e.g. Redis Cloud, AWS ElastiCache, or a separate Redis operator).

## Apply

```bash
# Update k8s/configmap.yaml with your values first
kubectl apply -f k8s/
```

## What's Deployed

| Manifest | Resource | Description |
|---|---|---|
| `configmap.yaml` | ConfigMap | Non-secret configuration (Redis host, embed URL, ports) |
| `orchestrator-deployment.yaml` | Deployment | C++ orchestrator; 2 replicas by default |
| `orchestrator-service.yaml` | Service | ClusterIP on port 8080 |
| `python-sidecar-deployment.yaml` | Deployment | Embedding sidecar; 2 replicas |
| `python-sidecar-service.yaml` | Service | ClusterIP on port 8001 |

## Secrets

The `OPENAI_API_KEY` should be injected as a Kubernetes Secret, not stored in the ConfigMap:

```bash
kubectl create secret generic lettucecache-secrets \
  --from-literal=OPENAI_API_KEY=sk-...
```

Reference it in the Deployment:

```yaml
env:
  - name: OPENAI_API_KEY
    valueFrom:
      secretKeyRef:
        name: lettucecache-secrets
        key: OPENAI_API_KEY
```

## FAISS Index Persistence

The FAISS index should be backed by a `PersistentVolumeClaim` so it survives pod restarts:

```yaml
volumes:
  - name: faiss-storage
    persistentVolumeClaim:
      claimName: faiss-pvc
volumeMounts:
  - name: faiss-storage
    mountPath: /data
```

Without a PVC, the index is rebuilt from scratch on every pod restart (all entries re-entered via the async write path as queries come in).

## Readiness Probe

The orchestrator's readiness probe hits `GET /health` and only passes when both Redis and the embedding sidecar are reachable:

```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
```

The Python sidecar has its own readiness probe on `GET /health` at port 8001.

## Scaling

The orchestrator is stateless (no session state; FAISS is in-memory per replica). Horizontal scaling works, but each replica has an independent FAISS index — entries cached on one pod won't be found by another.

For multi-replica deployments, consider:

- Using sticky sessions at the load balancer level (session affinity by `user_id` or `session_id`)
- Externalising FAISS to a shared vector store (e.g. Qdrant, Weaviate)

The L1 Redis cache is shared across all replicas automatically.
