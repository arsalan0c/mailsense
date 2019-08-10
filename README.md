# mailsense

> Subscribes to Gmail events to obtain new emails, performs sentiment analysis and then labels them.
![](demo.gif)

## Start
1. clone this repo: `git clone --recurse-submodules https://github.com/arsalanc-v2/mailsense.git`
2. `cd mailsense`
3. Register a Google Cloud project with a Pub/Sub subscription here: https://console.cloud.google.com/<br/>
Note the project name, subscription name and topic name (last section of the ids)
4. Download the Pub/Sub API credentials file from the cloud console
5. Set Google Application Credentials: `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/pubsub_credentials.json`
6. Register your project with the Gmail API
7. Download the Gmail API credentials file from the cloud console
8. `pip install -r requirements.txt`
9. Begin: `python3 mail/src/subscriber.py -p projectname -s subscriptionname -t topicname -cp /path/to/gmail_credentials.json -tp token.pickle -mt fastai -ma "{'model_dir':'./mail/sample', 'model_name':'textclassifier.pkl'}"`

On first run, you will be prompted to sign in and give the application access to view and modify your email.
  
## Scoring
Sentiment analysis is performed on an email's subject and a snippet of its body. 

There are 3 possible predicted values:
* `positive`
* `neutral`
* `negative`

The subject and body snippet are given the following scores according to their predicted sentiment:
* `positive`: `1`,
* `neutral`: `0`,
* `negative`: `-1`

They are then weighted with the subject given a weight of `0.3` and the body snippet given a weight of `0.7`.

The weighted scores for the subject and body snippet are summed to determine the overall sentiment of the email:
* `>= 0.5`: `positive`
* `< 0.5 && > -0.5`: `neutral`
* `<= -0.5`: `negative`

Table of outcomes:

Subject Sentiment | Body Snippet Sentiment | Overall Score | Overall Sentiment
----------------- | ---------------------- | ------------- | -----------------
positive          | positive               | 1.0           | positive
positive          | neutral                | 0.3           | neutral
positive          | negative               | -0.4          | neutral
neutral           | positive               | 0.7           | positive
neutral           | neutral                | 0.0           | neutral
neutral           | negative               | -0.7          | negative
negative          | positive               | 0.4           | neutral
negative          | neutral                | -0.3          | neutral
negative          | negative               | -1.0          | negative

## Models
The following sentiment analysis models are supported:

* [fastai text classifier](https://docs.fast.ai/tutorial.data.html#Classification). Usage: `-mt fastai -ma "{'model_dir': 'Directory of the saved fastai model', 'model_name': 'File name of the saved fastai model}"`
* [textblob Naive Bayes](https://textblob.readthedocs.io/en/dev/advanced_usage.html#sentiment-analyzers). Usage: `-mt textblob`
* [nltk sentiment intensity analysis](https://www.nltk.org/api/nltk.sentiment.html). Usage: `-mt nltk`

The latter two models are not finetuned on any additional data and are used as imported.<br/>
Each of the models has an inference script:<br/>
Eg. `python3 textblob/src/textblob_inference.py -t "The spice must flow"` prints `neutral`

### Fastai
A trained fastai model is provided at `mail/sample/textclassifier.pkl`.
It was trained on [EmoBank](https://github.com/JULIELab/EmoBank) and [Sentiment Labelled Sentences](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences). The one cycle policy was used in training to speed up training with a greater learning rate.

[DVC](https://dvc.org/) was used to benchmark the training experiments with different model hyperparameters.<br/>
See the pipeline: `dvc pipeline show fastai/dvc/evaluate_fastai.dvc --ascii`<br/>
Reproduce the pipeline: `dvc repro fastai/dvc/evaluate_fastai.dvc`

## Metrics
Sqlite (`mailsense.db`) is used to keep track of sentiment information:
* label assigned to each email and its datetime (without email text)

Run `python3 mail/src/metrics.py` to print some statistics to the console.

## Logs
The operations in `mail.py` are logged and output to `mail/logs/mail.log`.

Its general format is:

`{description of operation} {(if applicable) specific info for operation, eg. sentiment label} {(if applicable) message with history id: {history id of a Gmail subscriber message}}`
