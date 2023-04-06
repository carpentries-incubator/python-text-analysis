# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Custom dataset loading script for Library Carpentries"""

import os

import datasets

logger = datasets.logging.get_logger(__name__)

##Carpentries Note- This is where I got this dataset.
_CITATION = """\
@inproceedings{tjong-kim-sang-de-meulder-2003-introduction,
    title = "T-NER: An All-Round Python Library for Transformer-based Named Entity Recognition",
    author = "Asahi Ushio and Jose Camacho-Collados",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    year = "2021",
    url = "https://aclanthology.org/2021.eacl-demos.7",
    pages = "53--62",
}
"""

_DESCRIPTION = """\
A set of restaurant reviews and labels for entities for Amenities, Cuisine, Dish, Hours, Location, Price, Rating and Restaurant Name. For library carpentries.
"""

##Carpentries Note- We're changing the file names to match our dataset. You can also specify relative paths for data if you don't like requiring a URL. You'll see where this is downloaded later on.
_URL = "https://github.com/carpentries-incubator/python-text-analysis/raw/gh-pages/data/mit_restaurant.zip"
_TRAINING_FILE = "restauranttrain.bio"
_DEV_FILE = "restaurantvalid.bio"
_TEST_FILE = "restauranttest.bio"

##Carpentries Note- Find and replace on the conll2003 name, just to match the name of our python file.
class mit_restaurantsConfig(datasets.BuilderConfig):
    """BuilderConfig for mit_restaurants"""

    def __init__(self, **kwargs):
        """BuilderConfig formit_restaurants.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(mit_restaurantsConfig, self).__init__(**kwargs)


class mit_restaurants(datasets.GeneratorBasedBuilder):
    """mit_restaurants dataset."""

    BUILDER_CONFIGS = [
        ##Carpentries Note- Should be self-explainatory why this was changed
        mit_restaurantsConfig(name="mit_restaurants", version=datasets.Version("1.0.0"), description="MIT Dataset for restaurants"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                    ##Carpentries Note- Change these labels based on your project.
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-Amenity",
                                "I-Amenity",
                                "B-Cuisine",
                                "I-Cuisine",
                                "B-Dish",
                                "I-Dish",
                                "B-Hours",
                                "I-Hours",
                                "B-Location",
                                "I-Location",
                                "B-Price",
                                "I-Price",
                                "B-Rating",
                                "I-Rating",
                                "B-Restaurant_Name",
                                "I-Restaurant_Name",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://www.aclweb.org/anthology/W03-0419/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(_URL)
        data_files = {
            "train": os.path.join(downloaded_file, _TRAINING_FILE),
            "dev": os.path.join(downloaded_file, _DEV_FILE),
            "test": os.path.join(downloaded_file, _TEST_FILE),
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    ##Carpentries Note- mit_restaurants tokens are tab delimited, not space delimited
                    splits = line.split("\t")
                    ##Carpentries Note- We removed the POS and Chunk tags. We also had to modify the token and ner_tags index position since they happen to be reversed in our dataset vs conll.
                    ner_tags.append(splits[0])
                    tokens.append(splits[1].rstrip())
            # last example
            if tokens:
                yield guid, {
                    ##Carpentries- Notice we took out the other tags.
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
