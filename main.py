import wandb

from pprint import pprint
from inteledu import listener
from inteledu.datahub import DataHub
from inteledu.models.static.neural import NCDM

wandb.init(
    project="test inteledu"
)

listener.update(print)

datahub = DataHub("datasets/Math2")
datahub.random_split(source="total", to=["train", "test"])
print("Number of response logs {}".format(len(datahub)))

ncdm = NCDM(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
ncdm.build()
ncdm.train(datahub, "train", "test", valid_metrics=['auc', 'ap'])
test_results = ncdm.score(datahub, "test", metrics=["acc", "doa"])
pprint(test_results)
