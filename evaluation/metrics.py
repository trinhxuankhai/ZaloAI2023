from torchmetrics import Metric
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision
import torch
import torch.nn.functional as F

class ZaloMetric(Metric):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.inception_v3(pretrained=True)
        self.model.eval()
        self.model.fc = torch.nn.Identity()

        self.resize = torchvision.transforms.Resize(size=[299, 299])

        self.psnr = PeakSignalNoiseRatio(data_range=255.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=255.0)
        self.fid = FrechetInceptionDistance(feature=2048)

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("psnr_acc", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("ssim_acc", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("cosine_acc", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        #self.add_state("fid_acc", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds:torch.Tensor, target:torch.Tensor):
        #preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.total += preds.shape[0]
        self.psnr_acc += self.psnr(preds.type(torch.float32), target.type(torch.float32))
        self.ssim_acc += self.ssim(preds.type(torch.float32), target.type(torch.float32))
        self.fid.update(preds.type(torch.uint8), real=False)
        self.fid.update(target.type(torch.uint8), real=True)
        self.cosine_acc += self.consine_similarity(preds.type(torch.float32), target.type(torch.float32))

    def consine_similarity(self, preds, target):
        preds, target = self.resize(preds), self.resize(target)
        with torch.no_grad():
            preds_feature, target_feature = self.model(preds), self.model(target)
            similarity = F.cosine_similarity(preds_feature, target_feature).mean()
        return similarity

    def compute(self, reset=False):
        fid_acc = self.fid.compute()
        if reset:
            self.fid.reset()
        return 0.25*(self.psnr_acc + self.ssim_acc + self.cosine_acc) / self.total + fid_acc/4