# BERT for NER
This project is a NER based on Google [BERT](https://arxiv.org/pdf/1810.04805.pdf).

## Install BNER
To install the BNER package, run:
```text
pip install -e .
```

## Prepare your Google Cloud Platform environment
Run the following commands to set the proper project and zone:
```text
gcloud config set project <your-project>
gcloud config set compute/zone us-central1
```

Now you have to authorize the TPU to have access to ML-Engine. First get the service name of the 
TPU:
```text
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    https://ml.googleapis.com/v1/projects/<your-project>:getConfig
```

The output will look something like this:
```json
{
  "serviceAccount": "service-380211896040@cloud-ml.google.com.iam.gserviceaccount.com",
  "serviceAccountProject": "473645424018",
  "config": {
    "tpuServiceAccount": "service-473645424018@cloud-tpu.iam.gserviceaccount.com"
  }
}
```

Once you have the service name you have to set some authorization:
```text
gcloud projects add-iam-policy-binding <your-project>
    --member serviceAccount:<tpu-service> \
    --role roles/ml.serviceAgent
```

Next, you have to create the bucket that will contain the models and the data and set the 
authorizations:
```text
gsutil mb -c regional -l us-central1 gs://<bucket-name>
gsutil -m acl ch -r -u <tpu-service>:O gs://<bucket-name>
```
## Dataset format
To properly train the NER your dataset has to be in [CoNLL2003 format](https://www.clips.uantwerpen.be/conll2003/ner/).
The training set has to be named `train.conll` and the test set `test.conll`.

## Usage
NER entry point:
```text
# python -m bner.task --helpfull

USAGE: bner/task.py [flags]
flags:

bner.task.py:
  --adam_epsilon: Epsilon for Adam optimizer.
    (default: '1e-08')
    (a number)
  --batch_size: Total batch size for training.
    (default: '32')
    (an integer)
  --data_dir: The input data dir. Should contain the .conll files (or other data files) for the task.
  --epochs: Total number of training epochs to perform.
    (default: '3')
    (an integer)
  --learning_rate: Initial learning rate for Adam.
    (default: '5e-05')
    (a number)
  --max_seq_length: The maximum total input sentence length after tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.
    (default: '128')
    (an integer)
  --num_tpu_cores: Total number of TPU cores to use.
    (default: '8')
    (an integer)
  --output_dir: The output directory where the model checkpoints will be written.
  --tpu: The Cloud TPU to use for training. This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.
  --warmup_proportion: Proportion of training to perform linear learning rate warmup for.
    (default: '0.0')
    (a number)
  --weight_decay: Weight deay if we apply some.
    (default: '0.0')
    (a number)

absl.app:
  -?,--[no]help: show this help
    (default: 'false')
  --[no]helpfull: show full help
    (default: 'false')
  --[no]helpshort: show this help
    (default: 'false')
  --[no]helpxml: like --helpfull, but generates XML output
    (default: 'false')
  --[no]only_check_args: Set to true to validate args and exit.
    (default: 'false')
  --[no]pdb_post_mortem: Set to true to handle uncaught exceptions with PDB post mortem.
    (default: 'false')
  --profile_file: Dump profile information to a file (for python -m pstats). Implies --run_with_profiling.
  --[no]run_with_pdb: Set to true for PDB debug mode
    (default: 'false')
  --[no]run_with_profiling: Set to true for profiling the script. Execution will be slower, and the output format might change over time.
    (default: 'false')
  --[no]use_cprofile_for_profiling: Use cProfile instead of the profile module for profiling. This has no effect unless --run_with_profiling is set.
    (default: 'true')

absl.logging:
  --[no]alsologtostderr: also log to stderr?
    (default: 'false')
  --log_dir: directory to write logfiles into
    (default: '')
  --[no]logtostderr: Should only log to stderr?
    (default: 'false')
  --[no]showprefixforinfo: If False, do not prepend prefix to info messages when it's logged to stderr, --verbosity is set to INFO level, and python logging is used.
    (default: 'true')
  --stderrthreshold: log messages at this level, or more severe, to stderr in addition to the logfile.  Possible values are 'debug', 'info', 'warning', 'error', and 'fatal'.  Obsoletes
    --alsologtostderr. Using --alsologtostderr cancels the effect of this flag. Please also note that this flag is subject to --verbosity and requires logfile not be stderr.
    (default: 'fatal')
  -v,--verbosity: Logging verbosity level. Messages logged at this level or lower will be included. Set to 1 for debug logging. If the flag was not set or supplied, the value will be changed
    from the default of -1 (warning) to 0 (info) after flags are parsed.
    (default: '-1')
    (an integer)

absl.testing.absltest:
  --test_random_seed: Random seed for testing. Some test frameworks may change the default value of this flag between runs, so it is not appropriate for seeding probabilistic tests.
    (default: '301')
    (an integer)
  --test_randomize_ordering_seed: If positive, use this as a seed to randomize the execution order for test cases. If "random", pick a random seed to use. If 0 or not set, do not randomize
    test case execution order. This flag also overrides the TEST_RANDOMIZE_ORDERING_SEED environment variable.
  --test_srcdir: Root of directory tree where source files live
    (default: '')
  --test_tmpdir: Directory for temporary testing files
    (default: '/tmp/absl_testing')
  --xml_output_file: File to store XML test results
    (default: '')

tensorflow.python.ops.parallel_for.pfor:
  --[no]op_conversion_fallback_to_while_loop: If true, falls back to using a while loop for ops for which a converter is not defined.
    (default: 'false')

absl.flags:
  --flagfile: Insert flag definitions from the given file into the command line.
    (default: '')
  --undefok: comma-separated list of flag names that it is okay to specify on the command line even if the program does not define a flag with that name.  IMPORTANT: flags in this list that
    have arguments MUST use the --flag=value format.
    (default: '')
```

## Training on Google ML-Engine with TPU
You have to properly set the environment variables:
```text
STAGING_BUCKET=gs://<bucket-name>
JOB_NAME=<job-name>
```
 
And finally run the remote training:
```text
gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket $STAGING_BUCKET \
    --module-name bner.task \
    --package-path bner \
    --config configurations/config_tpu.yaml \
    -- \
    --data_dir gs://<bucket-name>/datasets \
    --output_dir gs://<bucket-name>/models \
    --tpu=$TPU_NAME
```

### Training on TPU without Google ML-Engine
To train the model on TPU, you have to create a VM and a TPU instance. To know how to do that you
can follow this example in the [documentation](https://cloud.google.com/tpu/docs/tutorials/mnist).
To start the training you can run the following command line:
```text
python -m bner.task \
    --data_dir gs://<bucket-name>/datasets \
    --output_dir gs://<bucket-name>/models \
    --tpu=$TPU_NAME
```

### Training on CPU/GPU with Google ML-Engine
It is not advised to train this model on CPU/GPU because you will easily need several days of
training instead of hours. Nevertheless, if you want to train this model on ML-Engine without a
TPU the process is the same except the command line that should be:
```text
gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket $STAGING_BUCKET \
    --module-name bner.task \
    --package-path bner \
    --config configurations/config_gpu.yaml \
    -- \
    --data_dir gs://<bucket-name>/datasets \
    --output_dir gs://<bucket-name>/models
```

### Training on CPU/GPU without Google ML-Engine
Finally, you can also run the training on a CPU/GPU on any platform (local, AWS or others) by 
running the following command line:
```text
python -m bner.task \
    --data_dir datasets \
    --output_dir models
```

## Docker images

### The serving Docker image
To create the serving image, run the following commands:
```text
docker run -d --gpus all --name serving_base tensorflow/serving:latest-gpu
mkdir -p model/<model-name>/<version>
```

If your model is stored on GCS:
```text
gsutil -m cp -R <saved-model-location>/* model/<model-name>/<version>
```

Otherwise:
```text
cp -R <saved-model-location>/* model/<model-name>/<version>
```

Then:
```text
docker cp model/<model-name> serving_base:/models/<model-name>
docker commit --change "ENV MODEL_NAME <model-name>" \
    --change "ENV PATH $PATH:/usr/local/nvidia/bin" \ 
    --change "ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64" serving_base <image-name>
docker kill serving_base
docker rm serving_base
```


### The client Docker image
To create and push the Docker image, run:
```text
docker build --build-arg METADATA=<location> --build-arg MODEL_NAME=<model-name> \
    -t <image-name> --no-cache .
docker push <image-name>
```

The *METADATA* argument represents the location where the `metadata.pkl` file created during the
training is. By default the the value is `model/metadata.pkl`. The *MODEL_NAME* argument is
mandatory, it represents the name of your model handled by the serving image.

## Deploy BNER in Kubernetes
To deploy BNER in Kubernetes you have to create a cluster with GPUs. Here I will detail the
deployment for Google Cloud Platform but I suppose it should be something similar on AWS and 
other platforms, just be careful to create your own Kubernetes manifests from the ones in the
[k8s](k8s) folder.

First create the cluster:
```text
gcloud container clusters create bner-cluster \
    --accelerator type=nvidia-tesla-v100,count=1 \
    --zone europe-west4-a \
    --cluster-version 1.12.5 \
    --machine-type n1-highmem-2 \
    --num-nodes=1 \
    --node-version 1.12.5-gke.5
```

Next, connect your `kubectl` to this new cluster:
```text
gcloud container clusters get-credentials bner-cluster \
    --zone europe-west4-a \
    --project <your-project>
```

Give a role to your node. First, retrieve the node of the node:
```text
kubectl get nodes
```

And then apply a label to this node:
```text
kubectl label nodes <node-name> bner-role=ner
```

Install the NVIDIA drivers to each node of the cluster:
```text
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

Wait a bit until the drivers are properly installed. Install the Helm server-side components
(Tiller):
```text
kubectl create serviceaccount -n kube-system tiller
kubectl create clusterrolebinding tiller-binding \
    --clusterrole=cluster-admin \
    --serviceaccount kube-system:tiller
helm init --service-account tiller
```

Once tiller pod becomes ready, update chart repositories:
```text
helm repo update
```

Install cert-manager:
```text
helm install --name cert-manager --version v0.5.2 \
    --namespace kube-system stable/cert-manager
```

Now you have to set up Let's Encrypt. Run this to deploy the Issuer manifests:
```text
kubectl apply -f k8s/certificate-issuer.yaml
```

Install BNER:
```text
kubectl apply -f k8s/deploy.yaml
```

And finally the ingress:
```text
kubectl apply -f k8s/ingress.yaml
```

