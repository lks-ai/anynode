import { api } from '../../scripts/api.js';
import { app } from '../../scripts/app.js';
import { $el } from '../../scripts/ui.js';

function get_position_style(ctx, widget_width, y, node_height) {
  const MARGIN = 16;

  /* Create a transform that deals with all the scrolling and zooming */
  const elRect = ctx.canvas.getBoundingClientRect();
  const initialTransform = ctx.getTransform();

  const transform = new DOMMatrix()
    .scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
    .multiplySelf(initialTransform)
    .translateSelf(MARGIN, y - MARGIN / 1.5);

  return {
    position: 'absolute',
    transform: transform.toString(),
    transformOrigin: '0 0',
    maxWidth: `${widget_width - MARGIN * 2}px`,
    minHeight: `${node_height - 56 - MARGIN / 2.5}px`,
    maxHeight: `${node_height - 56 - MARGIN / 2.5}px`,    // we're assuming we have the whole height of the node
  };
}

export function create_show_code_widget(code = '# Waiting for code...', language = 'python', id = 0) {
  const listener = api.addEventListener(`any-node-show-code-${id}`, (event) => {
    const { code, control, language, unique_id } = event.detail;
    widget.value = control;
    widget.language = language;
    if (!app.graph.getNodeById(unique_id).widgets_values) {
      app.graph.getNodeById(unique_id).widgets_values = [];
    }
    app.graph.getNodeById(unique_id).widgets_values[1] = control;
    update_show_code_widget(code, language, unique_id);
  });

  let widget = {
    node: id,
    type: 'CODE',
    name: 'CODE',
    html: $el('section', { className: 'get-position-style' }),
    language: 'python',
    copy() {
      if (!widget.value) {
        return navigator.clipboard.writeText(code);
      }

      if (this.language === 'python') {
        return navigator.clipboard.writeText(widget.value.function);
      }

      if (this.language === 'json') {
        return navigator.clipboard.writeText(JSON.stringify(widget.value, null, 4));
      }
    },
    draw(ctx, node, widget_width, y) {
      Object.assign(this.html.style, get_position_style(ctx, widget_width, y, node.size[1]));
    },
    onRemoved() {
      api.removeEventListener(`any-node-show-code-${id}`, listener);
    },
  };

  const highlightedCode = hljs.highlight(code, { language }).value;
  widget.html.innerHTML = `
    <pre>
      <code id="any-node-show-code-${id}" class="language-python"></code>
      <button class="copy-button" aria-label="Copy code">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
      </button>
    </pre>
  `;
  widget.html.querySelector(`#any-node-show-code-${id}`).innerHTML = highlightedCode;
  const codeEl = widget.html.querySelector('code');
  const button = widget.html.querySelector('.copy-button');
  button.addEventListener('click',
    () => {
      widget.copy();
      button.classList.remove('flash');
      codeEl.classList.remove('flash');
      setTimeout(() => button.classList.add('flash') || codeEl.classList.add('flash'), 0);
    }
  );
  document.body.appendChild(widget.html);

  return widget;
}

export function setMiniumSize(node, width, height) {
  if (node.size[0] < width) {
    node.size[0] = width;
  }

  if (node.size[1] < height) {
    node.size[1] = height;
  }
}

export function update_show_code_widget (code = '# Waiting for code...', language = 'python', id = 0) {
  const el = document.getElementById(`any-node-show-code-${id}`);
  el.innerHTML = hljs.highlight(code + '\n\n', { language }).value;
  hljs.initLineNumbersOnLoad();
}
