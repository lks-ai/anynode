import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { config } from "./constants.js";

function createDisplayWidget(node, app, message) {
  const outputWidget = ComfyWidgets["STRING"](
    node,
    "display_text",
    ["STRING", { multiline: true }],
    app
  ).widget;
  outputWidget.inputEl.readOnly = true;
  outputWidget.inputEl.style.opacity = 0.6;
  outputWidget.value = message.text.join("");
  return outputWidget;
}

function removeDisplayWidget(node) {
  const insertIndex = node.widgets.findIndex((w) => w.name === "display_text");
  if (insertIndex !== -1) {
    for (let i = insertIndex; i < node.widgets.length; i++) {
      node.widgets[i].onRemove?.();
    }
    node.widgets.length = insertIndex;
  }
}

function fitWidgetInNode(node, app) {
  requestAnimationFrame(() => {
    const computedSize = node.computeSize();
    if (computedSize[0] < node.size[0]) {
      computedSize[0] = node.size[0];
    }
    if (computedSize[1] < node.size[1]) {
      computedSize[1] = node.size[1];
    }
    node.onResize?.(computedSize);
    app.graph.setDirtyCanvas(true, false);
  });
}

const AnyNodeExtension = {
  name: "AnyNode",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData?.name && config.nodeBackendNames.includes(nodeData.name)) {
      const constructorPrototype = nodeType.prototype;
      const originalOnExecuted = constructorPrototype.onExecuted;

      constructorPrototype.onExecuted = function (message) {
        const node = this;
        originalOnExecuted?.apply(node, arguments);
        removeDisplayWidget(node);
        createDisplayWidget(node, app, message);
        fitWidgetInNode(node, app);
      };
    }
  },
};

app.registerExtension(AnyNodeExtension);
