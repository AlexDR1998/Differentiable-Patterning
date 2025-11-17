# Namespace
namespace='eidf151ns'

# Run yml file
run_file="/home/eidf151/eidf151/arichardson/Differentiable-Patterning/transfer.yml"

# Create run job
kubectl -n $namespace create -f $run_file

# Get run job name
job_name='ar-dp-job'

# Get the pod name of the pod attached to the run job
pod_name=$(kubectl -n $namespace get pod -l job-name=$job_name -o jsonpath="{.items[0].metadata.name}")
echo "Pod name: $pod_name"