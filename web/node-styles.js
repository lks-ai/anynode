export function applyStyles(node) {
  node.bgcolor = "#512222";

  (node.inputs || []).forEach(input => {
    if (input.name === 'control') {
      input.color_on = "#f495bf";
    }
  });

  (node.outputs || []).forEach(output => {
    if (output.name === 'control') {
      output.color_on = "#f495bf";
    }
  });

  (node.widgets || []).forEach(widget => {
    if (widget.type === 'customtext') {
      widget.inputEl.classList.add('any-node-input');
    }
  });
}
