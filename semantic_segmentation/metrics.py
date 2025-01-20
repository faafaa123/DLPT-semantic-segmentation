# External imports
import torch
from torcheval.metrics import MulticlassConfusionMatrix
import segmentation_models_pytorch as smp


class DiceScore(torch.nn.Module):
    """ """

    def __init__(self, num_classes, ignore_index: int | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = 1e-12
        self.metric = MulticlassConfusionMatrix(self.num_classes)

    def __call__(self, pred, target):
        """
        pred: NxHxW
        target: NxHxW
        """

        self.metric.reset()
        self.metric.update(pred.flatten(), target.flatten())
        conf_matrix = self.metric.compute()

        if self.ignore_index is not None:
            # set column values of ignore classes to 0
            conf_matrix[:, self.ignore_index] = 0
            # set row values of ignore classes to 0
            conf_matrix[self.ignore_index, :] = 0

        true_positive = torch.diag(conf_matrix)
        false_positive = torch.sum(conf_matrix, 0) - true_positive
        false_negative = torch.sum(conf_matrix, 1) - true_positive

        DSC = (2 * true_positive + self.eps) / (
            2 * true_positive + false_positive + false_negative + self.eps
        )

        return DSC


class IOU(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def __call__(self, pred, target):
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred, target, mode="multiclass", num_classes=self.num_classes
        )
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)
        return iou_score


if __name__ == "__main__":

    # Test the Dice score implementation

    from semantic_segmentation.utils import setup_system
    from semantic_segmentation.configuration import SystemConfig

    setup_system(SystemConfig())

    # Create ground truth data
    ground_truth = torch.zeros(1, 224, 224, dtype=torch.int64)
    ground_truth[:, 50:100, 50:100] = 1
    ground_truth[:, 50:150, 150:200] = 2

    # Generate torch tensor to check the solution
    prediction_prob = torch.zeros(1, 3, 224, 224).uniform_().softmax(dim=1)
    class_prediction = prediction_prob.argmax(dim=1)

    dice_coeff = DiceScore(num_classes=3)
    score = dice_coeff(pred=class_prediction, target=ground_truth)

    expected = torch.Tensor([0.47558773, 0.0892497, 0.14908088])
    torch.testing.assert_close(score, expected)
    print("score: ", score)
    print("expected: ", expected)
