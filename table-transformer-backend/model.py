from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse


class NewModel(
    LabelStudioMLBase,
):
    # change this to microsoft/table-transformer-structure-recognition for table structure prediction
    # or your own finetuned model
    model_name = "microsoft/table-transformer-structure-recognition"

    def setup(self, **kwargs):
        self.set("model_version", "table-transformer-v1")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"DEBUG: Received model_name = {self.model_name}")
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = TableTransformerForObjectDetection.from_pretrained(
            self.model_name
        ).to(self.device)

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> ModelResponse:
        print(f"Run prediction on {tasks}")
        predictions = []

        for task in tasks:
            # Get local path to the PNG image
            image_path = self.get_local_path(
                task["data"]["image"],
                task_id=task["id"],
                # ls_access_token=self.access_token,
            )
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            # Preprocess
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process predictions
            # Change the detection threshold accordingly
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=[image.size[::-1]], threshold=0.8
            )[0]

            # Convert to Label Studio JSON format
            width, height = image.size
            result = []
            for i in range(len(results["boxes"])):
                box = results["boxes"][i]
                label = results["labels"][i]
                score = results["scores"][i]

                # normalized bbox for Label Studio
                x, y, x2, y2 = box.tolist()
                result.append(
                    {
                        "from_name": "label",  # must match your labeling config
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "rectanglelabels": [
                                str(self.model.config.id2label[label.item()])
                            ],
                            "x": x / width * 100,
                            "y": y / height * 100,
                            "width": (x2 - x) / width * 100,
                            "height": (y2 - y) / height * 100,
                        },
                        "score": float(score),
                    }
                )

            predictions.append(
                {"result": result, "model_version": self.get("model_version")}
            )

        return ModelResponse(predictions=predictions)
