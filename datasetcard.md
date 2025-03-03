---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---

# Dataset Card for FEA-Bench

<!-- Provide a quick summary of the dataset. -->
A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation.

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

The FEA-Bench is a benchmark with a test set that contains 1,401 task instances from 83 Github repositories. This benchmark aims to evaluate the capabilities of repository-level incremental code development. The task instances are collected from Github pull requests, which have the purpose of new feature implementation. Each task instance includes the repo and the base commit sha256, and the PR number and the status of unit test.


- **Curated by:** the authors of the FEA-Bench paper: Wei Li, Xin Zhang, Zhongxin Guo, Shaoguang Mao and their collaborators.
- **Language(s) (NLP):** English
- **License:** Others; We list all licenses of involved github repositories in the last part.

<!-- - **Funded by [optional]:** {{ funded_by | default("[More Information Needed]", true)}}
- **Shared by [optional]:** {{ shared_by | default("[More Information Needed]", true)}} -->

<!-- ### Dataset Sources [optional] -->

<!-- Provide the basic links for the dataset. -->

<!-- - **Repository:** {{ repo | default("[More Information Needed]", true)}}
- **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}}
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}} -->

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

This dataset is designed to evaluate performances of LLMs on repository-level code development, which is a complicated software engineering task.

- Repository-level incremental code development: The FEA-Bench can be used to evaluate a model for the the capabilities of repository-level incremental code development. Success on this task is typically measured by achieving a high/low resolved ratio. The leaderboard will soon be published as a website.

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

Use scripts from FEA-Bench repo to get info for task instances and organize them into prompt, which can be used to LLMs' inference. Also, you can get info or use agents to directly solve the PRs with code changes.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

This dataset is not aimed at training for LLMs. You should not take the FEA-Bench as the training dataset to avoid contamination.

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->
An example:
```
{
    "instance_id": "huggingface__accelerate-270",
    "pull_number": 270,
    "repo": "huggingface/accelerate",
    "version": null,
    "base_commit": "515fcca9ed2b36c274c595dbdff75f1c2da635de",
    "environment_setup_commit": "08101b9dde2b1a9658c2e363e3e9f5663ba06073",
    "FAIL_TO_PASS": [
        "tests/test_state_checkpointing.py::CheckpointTest::test_can_resume_training",
        "tests/test_state_checkpointing.py::CheckpointTest::test_invalid_registration",
        "tests/test_state_checkpointing.py::CheckpointTest::test_with_scheduler"
    ],
    "PASS_TO_PASS": []
}
```

## Dataset Creation

### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

Implementing new features in repository-level codebases is a crucial application of code generation models. However, current benchmarks lack a dedicated evaluation framework for this capability. To fill this gap, we introduce FEA-Bench, a benchmark designed to assess the ability of large language models (LLMs) to perform incremental development within code repositories.

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

We collect pull requests from 83 GitHub repositories and use rule-based and intent-based filtering to construct task instances focused on new feature development. Each task instance containing code changes is paired with relevant unit test files to ensure that the solution can be verified. 

#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

Authors of 83 Github repositories list in the last part.

<!-- ### Annotations [optional] -->

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

<!-- #### Annotation process -->

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

<!-- {{ annotation_process_section | default("[More Information Needed]", true)}} -->

<!-- #### Who are the annotators? -->

<!-- This section describes the people or systems who created the annotations. -->

<!-- {{ who_are_annotators_section | default("[More Information Needed]", true)}} -->

#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

The dataset does not include any personal or sensitive information.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->
- The quantity of high-quality data suitable for repository-level incremental development is limited. High-quality and usable pull requests for new feature development are relatively scarce. Many repository-level code developments for implementing new functionalities were committed during the early stages of repositories, without going through the rigorous code review process typical of the open-source community, resulting in lower data quality that cannot be utilized. 
- Furthermore, the software's early-stage developments might not even have been conducted using the GitHub platform, posing a challenge for data collection and utilization.
- The repository-level incremental code development may not just include new feature implementation tasks.
- Only Python repositories are involved in FEA-Bench.
- The inference results of the task instances from the benchmark may contain code that is harmful to computer systems.


### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Evaluation by docker is recommended, just like SWE-bench. We will also publish a patch for SWE-bench to make it compatible for our tasks' evaluation.

<!-- ## Citation [optional] -->

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

To be appeared after publishing the ArXiv paper.

**APA:**

To be appeared after publishing the ArXiv paper.


## Dataset Card Contact

For further information or questions, please contact Xin Zhang (xinzhang3@microsoft.com).


## All involved Github repositories in the FEA-Bench

| Repo Name                         | License           | Topic                                      |
|-----------------------------------|-------------------|--------------------------------------------|
| astropy/astropy                   | BSD-3-Clause      | Scientific/Engineering::Astronomy          |
| django/django                     | BSD-3-Clause      | Internet::WWW/HTTP                         |
| matplotlib/matplotlib             | Other             | Scientific/Engineering::Visualization     |
| mwaskom/seaborn                   | BSD-3-Clause      | Scientific/Engineering::Visualization     |
| pallets/flask                     | BSD-3-Clause      | Internet::WWW/HTTP                         |
| pvlib/pvlib-python                | BSD-3-Clause      | Scientific/Engineering::Physics           |
| pydata/xarray                     | Apache-2.0        | Scientific/Engineering::Information Analysis |
| pydicom/pydicom                   | Others            | Scientific/Engineering::Medical Science Apps. |
| pylint-dev/astroid                | LGPL-2.1          | Software Development::Libraries           |
| pylint-dev/pylint                 | GPL-2.0           | Software Development::Quality Assurance   |
| pyvista/pyvista                   | MIT               | Scientific/Engineering::Information Analysis |
| scikit-learn/scikit-learn         | BSD-3-Clause      | Scientific/Engineering::Artificial Intelligence |
| sphinx-doc/sphinx                 | BSD-2-Clause      | Text Processing::Markup                   |
| sqlfluff/sqlfluff                 | MIT               | Software Development::Quality Assurance   |
| sympy/sympy                       | Others            | Scientific/Engineering::Mathematics       |
| Aider-AI/aider                    | Apache-2.0        | Software Development::Code Generators     |
| Cog-Creators/Red-DiscordBot       | GPL-3.0           | Communications::Chat                      |
| DLR-RM/stable-baselines3          | MIT               | Scientific/Engineering::Artificial Intelligence |
| EleutherAI/lm-evaluation-harness  | MIT               | Scientific/Engineering::Artificial Intelligence |
| Project-MONAI/MONAI               | Apache-2.0        | Scientific/Engineering::Medical Science Apps. |
| PyThaiNLP/pythainlp               | Apache-2.0        | Text Processing::Linguistic               |
| RDFLib/rdflib                     | BSD-3-Clause      | Software Development::Libraries           |
| Textualize/rich                   | MIT               | Software Development::Libraries           |
| Textualize/textual                | MIT               | Software Development::User Interfaces     |
| TileDB-Inc/TileDB-Py              | MIT               | Software Development::Libraries           |
| astronomer/astronomer-cosmos      | Apache-2.0        | Software Development::Build Tools         |
| atlassian-api/atlassian-python-api| Apache-2.0        | Internet::WWW/HTTP                        |
| aws-cloudformation/cfn-lint       | MIT-0             | Software Development::Quality Assurance   |
| aws-powertools/powertools-lambda-python | MIT-0       | Software Development::Libraries           |
| aws/sagemaker-python-sdk          | Apache-2.0        | Scientific/Engineering::Artificial Intelligence |
| biopragmatics/bioregistry         | MIT               | Scientific/Engineering::Bio-Informatics   |
| boto/boto3                        | Apache-2.0        | Software Development::Libraries           |
| boto/botocore                     | Apache-2.0        | Software Development::Libraries           |
| cocotb/cocotb                     | BSD-3-Clause      | Scientific/Engineering::Electronic Design Automation (EDA) |
| conan-io/conan                    | MIT               | Software Development::Build Tools         |
| deepset-ai/haystack               | Apache-2.0        | Scientific/Engineering::Artificial Intelligence |
| docker/docker-py                  | Apache-2.0        | Software Development::Libraries           |
| dpkp/kafka-python                 | Apache-2.0        | Software Development::Libraries           |
| embeddings-benchmark/mteb         | Apache-2.0        | Scientific/Engineering::Artificial Intelligence |
| facebookresearch/hydra            | MIT               | Software Development::Libraries           |
| fairlearn/fairlearn               | MIT               | Scientific/Engineering::Artificial Intelligence |
| falconry/falcon                   | Apache-2.0        | Internet::WWW/HTTP                        |
| google-deepmind/optax             | Apache-2.0        | Scientific/Engineering::Artificial Intelligence |
| googleapis/python-aiplatform      | Apache-2.0        | Scientific/Engineering::Artificial Intelligence |
| googleapis/python-bigquery        | Apache-2.0        | Internet::WWW/HTTP                        |
| gradio-app/gradio                 | Apache-2.0        | Scientific/Engineering::Human Machine Interfaces |
| graphql-python/graphene           | MIT               | Software Development::Libraries           |
| huggingface/accelerate            | Apache-2.0        | Scientific/Engineering::Artificial Intelligence |
| huggingface/datasets              | Apache-2.0        | Scientific/Engineering::Artificial Intelligence |
| huggingface/huggingface_hub       | Apache-2.0        | Scientific/Engineering::Artificial Intelligence |
| huggingface/pytorch-image-models  | Apache-2.0        | Software Development::Libraries           |
| huggingface/trl                   | Apache-2.0        | Scientific/Engineering::Artificial Intelligence |
| joblib/joblib                     | BSD-3-Clause      | Software Development::Libraries           |
| joke2k/faker                      | MIT               | Software Development::Testing             |
| lark-parser/lark                  | MIT               | Text Processing::Linguistic               |
| minio/minio-py                    | Apache-2.0        | Software Development::Libraries           |
| open-mmlab/mmengine               | Apache-2.0        | Utilities                                 |
| openvinotoolkit/datumaro          | MIT               | Scientific/Engineering::Image Processing  |
| pgmpy/pgmpy                       | MIT               | Scientific/Engineering::Artificial Intelligence |
| pre-commit/pre-commit             | MIT               | Software Development::Quality Assurance   |
| prometheus/client_python          | Apache-2.0        | System::Monitoring                        |
| prompt-toolkit/python-prompt-toolkit | BSD-3-Clause   | Software Development::User Interfaces     |
| pygments/pygments                 | BSD-2-Clause      | Software Development::Documentation       |
| pyocd/pyOCD                       | Apache-2.0        | Software Development::Debuggers           |
| pypa/hatch                        | MIT               | Software Development::Build Tools         |
| pyro-ppl/pyro                     | Apache-2.0        | Scientific/Engineering::Artificial Intelligence |
| python-hyper/h2                   | MIT               | Internet::WWW/HTTP                        |
| roboflow/supervision              | MIT               | Scientific/Engineering::Image Processing  |
| rytilahti/python-miio             | GPL-3.0           | Home Automation                           |
| saleweaver/python-amazon-sp-api   | MIT               | Internet::WWW/HTTP                        |
| scrapy/scrapy                     | BSD-3-Clause      | Software Development::Libraries           |
| scverse/scanpy                    | BSD-3-Clause      | Scientific/Engineering::Bio-Informatics   |
| slackapi/bolt-python              | MIT               | Communications::Chat                      |
| slackapi/python-slack-sdk         | MIT               | Communications::Chat                      |
| snowflakedb/snowflake-connector-python | Apache-2.0    | Software Development::Libraries           |
| softlayer/softlayer-python        | MIT               | Software Development::Libraries           |
| spec-first/connexion              | Apache-2.0        | Internet::WWW/HTTP                        |
| statsmodels/statsmodels           | BSD-3-Clause      | Scientific/Engineering::Information Analysis |
| tfranzel/drf-spectacular          | BSD-3-Clause      | Software Development::Documentation       |
| tobymao/sqlglot                   | MIT               | Database::Database Engines/Servers        |
| tornadoweb/tornado                | Apache-2.0        | Internet::WWW/HTTP                        |
| tortoise/tortoise-orm             | Apache-2.0        | Database::Front-Ends                      |
| wagtail/wagtail                   | BSD-3-Clause      | Internet::WWW/HTTP                        |
