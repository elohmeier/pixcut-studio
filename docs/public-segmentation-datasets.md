# Public data for cut-contour training

The required label is the **outer printed cut border of each sticker**, including
its white rim and attached decoration. General object masks stop at the subject.
They are useful for pretraining or for creating synthetic sticker sheets, but
they cannot replace reviewed examples from actual sticker sheets.

| Dataset | Relevant signal | Practical use | Terms / limitation |
| --- | --- | --- | --- |
| [AM-2k](https://github.com/JizhiziLi/GFM) | 2,000 animal photos with soft alpha mattes | First targeted follow-up: animal and fur edges; generate a white rim after alpha compositing | The [release agreement](https://jizhizili.github.io/files/gfm_datasets_agreements/AM-2k_Dataset_Release_Agreement.pdf) lists MIT terms, with original image copyright retained by its owners. The research framing does not say that the data is categorically noncommercial. |
| [P3M-10k](https://github.com/JizhiziLi/P3M) | High-resolution portrait alpha mattes, especially hair | Next targeted follow-up: pale hair; composite portraits into synthetic stickers | The [dataset release agreement](https://jizhizili.github.io/files/p3m_dataset_agreement/P3M-10k_Dataset_Release_Agreement.pdf) lists MIT terms, but original image owners retain copyright. Faces are blurred in the training images. |
| [Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) | 2.8 million instance masks across 350 classes | Broader subject diversity and known instance IDs, after the targeted tests | Annotations are CC BY 4.0. The project lists images as CC BY 2.0 but explicitly asks users to verify each image's license. Filter and keep attribution metadata. |
| [SA-V](https://github.com/facebookresearch/sam2/blob/main/sav_dataset/README.md) | Many instance masks with stable IDs across video frames | Optional generic instance-separation pretraining; sample sparse frames to avoid redundant videos | CC BY 4.0; far larger and less aligned with printed artwork than Open Images. |
| [DIS5K](https://github.com/xuebinqin/DIS) | Detailed high-resolution foreground contours | Private noncommercial training experiment: composite source objects into synthetic stickers with exact generated white rims | The [dataset terms](https://github.com/xuebinqin/DIS/blob/main/DIS5K-Dataset-Terms-of-Use.pdf) permit noncommercial research and educational use and prohibit redistribution. The code's Apache license does not cover the dataset. |
| [HRSOD](https://github.com/yi94code/HRSOD) / [UHRSD](https://github.com/iCVTEAM/PGNet) | 2,010 / 5,920 high-resolution saliency images | Possible edge pretraining if the foreground model is redesigned to use larger context | Their release pages do not provide clear separate data licenses; saliency masks still describe subjects, not print borders. Resolution alone does not help a 256-pixel tile model. |
| [DUTS](https://saliencydetection.net/duts/) | 10,553 train and 5,019 test images with saliency masks | Large, low-priority source for coarse foreground pretraining | Its website says annotation rights are reserved by the authors; lower border fidelity than alpha mattes. |
| [COCO](https://cocodataset.org/) / [LVIS](https://www.lvisdataset.org/) | Many category-labeled instances | Pretrain **which object belongs to which instance** or supply varied subjects for synthetic sheets | Object polygons omit sticker rims and attached decorations. Check image-level terms before use. |

No source in this review supplies the exact multi-sticker sheet and white-border
cut targets needed here. The matched DIS5K experiment improved held-out
synthetic-source scores but barely changed real-sheet results relative to extra
training without public data. Its [official release](https://github.com/xuebinqin/DIS)
also says its first version has few human and animal images. AM-2k and P3M are
more targeted tests of the animal and hair domains in these sheets. Even they
cannot teach which decorations share one sticker or where two white margins
touch. Keep actual sticker sheets for development and a separate untouched
sheet-level test.

The current synthetic scene generator already composites reviewed stickers.
The AM-2k adapter composites alpha mattes on white, generates an outer rim and
records the source IDs. It takes only original/mask pairs from the official
training split; its validation split stays untouched. For any later Open Images
adapter, retain image IDs, license URLs and attribution. Public source images
and masks remain in a private local cache; they are not copied into this repo.

Evaluate every new source against the same no-public control on the actual
sticker sheets. Use per-instance matching, extra/missing cut paths, overlaps,
and contour distance on reviewed outer borders. A synthetic-source score alone
does not justify switching the browser detector.
