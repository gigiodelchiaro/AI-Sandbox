const canvas = document.getElementById('nnCanvas');
const ctx = canvas.getContext('2d');



function ColorAlphaBlend(lowColor, highColor, alpha) {
    const blend = (c1, c2, a) => Math.floor(c1 * (1 - a) + c2 * a);
    return `rgba(${blend(lowColor.r, highColor.r, alpha)}, ${blend(lowColor.g, highColor.g, alpha)}, ${blend(lowColor.b, highColor.b, alpha)}, 1)`;
}

function drawLineEx(start, end, thick, color) {
    ctx.lineWidth = thick;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
}

function drawCircle(x, y, radius, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
}

function nnRenderRaylib(nn, rx, ry, rw, rh) {
    const lowColor = { r: 255, g: 0, b: 255, a: 1 };
    const highColor = { r: 0, g: 255, b: 0, a: 1 };

    const neuronRadius = rh * 0.04;
    const layerBorderVPad = 50;
    const layerBorderHPad = 50;
    const nnWidth = rw - 2 * layerBorderHPad;
    const nnHeight = rh - 2 * layerBorderVPad;
    const nnX = rx + rw / 2 - nnWidth / 2;
    const nnY = ry + rh / 2 - nnHeight / 2;
    const archCount = nn.count + 1;
    const layerHPad = nnWidth / archCount;

    for (let l = 0; l < archCount; ++l) {
        const layerVPad1 = nnHeight / nn.as[l].cols;
        for (let i = 0; i < nn.as[l].cols; ++i) {
            const cx1 = nnX + l * layerHPad + layerHPad / 2;
            const cy1 = nnY + i * layerVPad1 + layerVPad1 / 2;
            if (l + 1 < archCount) {
                const layerVPad2 = nnHeight / nn.as[l + 1].cols;
                for (let j = 0; j < nn.as[l + 1].cols; ++j) {
                    const cx2 = nnX + (l + 1) * layerHPad + layerHPad / 2;
                    const cy2 = nnY + j * layerVPad2 + layerVPad2 / 2;
                    const value = sigmoid(nn.ws[l][j][i]);
                    const color = ColorAlphaBlend(lowColor, highColor, value);
                    const thick = rh * 0.004;
                    drawLineEx({ x: cx1, y: cy1 }, { x: cx2, y: cy2 }, thick, color);
                }
            }
            if (l > 0) {
                const alpha = sigmoid(nn.bs[l - 1][0][i]);
                const color = ColorAlphaBlend(lowColor, highColor, alpha);
                drawCircle(cx1, cy1, neuronRadius, color);
            } else {
                drawCircle(cx1, cy1, neuronRadius, 'gray');
            }
        }
    }
}
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
function create_nn(arch) {
    const nn = {
        count: arch.length - 1,
        as: [],
        ws: [],
        bs: []
    };

    for (let i = 0; i < nn.count + 1; i++) {
        nn.as.push({ cols: arch[i] });
    }

    for (let i = 0; i < nn.count; i++) {
        const numRows = arch[i + 1];
        const numCols = arch[i];
        const weights = [];
        const biases = [];

        for (let j = 0; j < numRows; j++) {
            const row = [];
            const biasRow = [];
            for (let k = 0; k < numCols; k++) {
                row.push(Math.random() - 0.5); // Random weight initialization between -0.5 and 0.5
            }
            weights.push(row);
            biasRow.push(Math.random() - 0.5); // Random bias initialization between -0.5 and 0.5
            biases.push(biasRow);
        }

        nn.ws.push(weights);
        nn.bs.push(biases);
    }

    return nn;
}

function feedForward(nn, input) {
    let activations = input;

    for (let i = 0; i < nn.count; i++) {
        const weights = nn.ws[i];
        const biases = nn.bs[i];
        const layerActivations = [];

        for (let j = 0; j < weights.length; j++) {
            let sum = 0;
            for (let k = 0; k < weights[j].length; k++) {
                sum += weights[j][k] * activations[k];
            }
            sum += biases[j][0]; // Adding bias
            layerActivations.push(sigmoid(sum));
        }

        activations = layerActivations;
    }

    return activations;
}

