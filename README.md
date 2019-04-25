# BERT for NER
This project is a NER based on Google [BERT](https://arxiv.org/pdf/1810.04805.pdf).

## Install BNER
To install the BNER package, run:
```text
pip install -e bner .
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
gsutil acl ch -u <tpu-service>:READER gs://<bucket-name>
gsutil acl ch -u <tpu-service>:WRITER gs://<bucket-name>
```
## Dataset format
To properly train the NER your dataset has to be in [CoNLL2003 format](https://www.clips.uantwerpen.be/conll2003/ner/).
The training set has to be named `train.conll` and the test set `test.conll`.

## Usage
NER entry point:
```text
# python -m bner.ner.task --helpfull

USAGE: bner/ner/task.py [flags]
flags:

absl.logging:
  --[no]alsologtostderr: also log to stderr?
    (default: 'false')
  --log_dir: directory to write logfiles into
    (default: '')
  --[no]logtostderr: Should only log to stderr?
    (default: 'false')
  --[no]showprefixforinfo: If False, do not prepend prefix to info messages when it's logged to stderr, --verbosity is set to INFO level, and python logging is used.
    (default: 'true')
  --stderrthreshold: log messages at this level, or more severe, to stderr in addition to the logfile.  Possible values are 'debug', 'info', 'warning', 'error', and 'fatal'.  Obsoletes --alsologtostderr. Using --alsologtostderr cancels the effect of this flag.
    Please also note that this flag is subject to --verbosity and requires logfile not be stderr.
    (default: 'fatal')
  -v,--verbosity: Logging verbosity level. Messages logged at this level or lower will be included. Set to 1 for debug logging. If the flag was not set or supplied, the value will be changed from the default of -1 (warning) to 0 (info) after flags are parsed.
    (default: '-1')
    (an integer)

bert.run_classifier:
  --bert_config_file: The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.
  --data_dir: The input data dir. Should contain the .tsv files (or other data files) for the task.
  --[no]do_eval: Whether to run eval on the dev set.
    (default: 'false')
  --[no]do_lower_case: Whether to lower case the input text. Should be True for uncased models and False for cased models.
    (default: 'true')
  --[no]do_predict: Whether to run the model in inference mode on the test set.
    (default: 'false')
  --[no]do_train: Whether to run training.
    (default: 'false')
  --eval_batch_size: Total batch size for eval.
    (default: '8')
    (an integer)
  --gcp_project: [Optional] Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically detect the GCE project from metadata.
  --init_checkpoint: Initial checkpoint (usually from a pre-trained BERT model).
  --iterations_per_loop: How many steps to make in each estimator call.
    (default: '1000')
    (an integer)
  --learning_rate: The initial learning rate for Adam.
    (default: '5e-05')
    (a number)
  --master: [Optional] TensorFlow master URL.
  --max_seq_length: The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.
    (default: '128')
    (an integer)
  --num_tpu_cores: Only used if `use_tpu` is True. Total number of TPU cores to use.
    (default: '8')
    (an integer)
  --num_train_epochs: Total number of training epochs to perform.
    (default: '3.0')
    (a number)
  --output_dir: The output directory where the model checkpoints will be written.
  --predict_batch_size: Total batch size for predict.
    (default: '8')
    (an integer)
  --save_checkpoints_steps: How often to save the model checkpoint.
    (default: '1000')
    (an integer)
  --task_name: The name of the task to train.
  --tpu_name: The Cloud TPU to use for training. This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.
  --tpu_zone: [Optional] GCE zone where the Cloud TPU is located in. If not specified, we will attempt to automatically detect the GCE project from metadata.
  --train_batch_size: Total batch size for training.
    (default: '32')
    (an integer)
  --[no]use_tpu: Whether to use TPU or GPU/CPU.
    (default: 'false')
  --vocab_file: The vocabulary file that the BERT model was trained on.
  --warmup_proportion: Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.
    (default: '0.1')
    (a number)

  tensorflow.python.platform.app:
  -h,--[no]help: show this help
    (default: 'false')
  --[no]helpfull: show full help
    (default: 'false')
  --[no]helpshort: show this help
    (default: 'false')

  tensorflow_hub.resolver:
  --tfhub_cache_dir: If set, TF-Hub will download and cache Modules into this directory. Otherwise it will attempt to find a network path.

  absl.flags:
  --flagfile: Insert flag definitions from the given file into the command line.
    (default: '')
  --undefok: comma-separated list of flag names that it is okay to specify on the command line even if the program does not define a flag with that name.  IMPORTANT: flags in this list that have arguments MUST use the --flag=value format.
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
gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket $STAGING_BUCKET \
    --module-name bner.ner.task \
    --package-path bner \
    --config configurations/config_tpu.yaml \
    -- \
    --data_dir gs://<bucket-name>/datasets \
    --output_dir gs://<bucket-name>/models \
    --tfhub_cache_dir gs://<bucket-name>/tfhub \
    --learning_rate 2e-5 \
    --train_batch_size 256 \
    --num_train_epochs 4.0 \
    --use_tpu True
```

### Training on TPU without Google ML-Engine
To train the model on TPU, you have to create a VM and a TPU instance. To know how to do that you
can follow this example in the [documentation](https://cloud.google.com/tpu/docs/tutorials/mnist).
To start the training you can run the following command line:
```text
python bner/ner/task.py \
    --data_dir gs://<bucket-name>/datasets \
    --output_dir gs://<bucket-name>/models \
    --tfhub_cache_dir gs://<bucket-name>/tfhub \
    --learning_rate 2e-5 \
    --train_batch_size 256 \
    --num_train_epochs 4.0 \
    --use_tpu True \
    --tpu_name=$TPU_NAME
```

### Training on CPU/GPU with Google ML-Engine
It is not advised to train this model on CPU/GPU because you will easily need several days of
training instead of hours. Nevertheless, if you want to train this model on ML-Engine without a
TPU the process is the same except the command line that should be:
```text
gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket $STAGING_BUCKET \
    --module-name bner.ner.task \
    --package-path bner \
    --config configurations/config_gpu.yaml \
    -- \
    --data_dir gs://<bucket-name>/datasets \
    --output_dir gs://<bucket-name>/models \
    --num_train_epochs 4.0 \
    --learning_rate 2e-5
```

### Training on CPU/GPU without Google ML-Engine
Finally, you can also run the training on a CPU/GPU on any platform (local, AWS or others) by 
running the following command line:
```text
python bner/ner/task.py \
    --data_dir datasets \
    --output_dir models \
    --num_train_epochs 4.0 \
    --learning_rate 2e-5
```

## Docker images

### The serving Docker image
To create the serving image, run the following commands:
```text
docker run -d --runtime=nvidia --name serving_base tensorflow/serving:latest-gpu
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

