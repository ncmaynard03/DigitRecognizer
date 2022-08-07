const mmap = math.map;
const rand = math.random;
const transp = math.transpose;
const mat = math.matrix;
const e = math.evaluate;
const sub = math.subtract;
const sqr = math.square;
const sum = math.sum;

class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes, learningRate, wih, who) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.learningRate = learningRate;

        this.wih = wih || sub(mat(rand([hiddenNodes, inputNodes])), 0.5);
        this.who = who || sub(mat(rand([outputNodes, hiddenNodes])), 0.5);

        this.act = (matrix) => mmap(matrix, (x) => 1 / (1 + Math.exp(-x)));
    }

    cache = { loss: [] };

    static normalizeData = (data) => { /*...*/};

    forward = (input) => {   
        const wih = this.wih;
        const who = this.who;
        const act = this.act;

        input = transp(mat([input]));

        const h_in = e("wih * input", {wih, input});
        const h_out = act(h_in);

        const o_in = e("who * h_out", {who, h_out});
        const actual = act(o_in);

        this.cache.input = input;
        this.cache.h_out = h_out;
        this.cache.actual = actual;
 
        return actual;
    };

    backward = (input, target) => { 
        const who = this.who;
        const input = this.cache.input;
        const h_out = this.cache.h_out;
        const actual = this.cache.actual;

        target = transp(mat([target]));

        const dEdA = sub(target, actual);

        const o_dAdZ = e("actual .* (1 - actual", {
            actual,
        });

        const dwho = e("(dEdA .* o_dAdZ) * h_out'", {
            dEdA,
            o_dAdZ,
            h_out,
        });

        const h_err = e("who' * (dEdA .* o_dAdZ)", {
            who, 
            dEdA,
            o_dAdZ,
        });

        const h_dAdZ = e("h_out .* (1-h_out)", {
            h_out,
        });

        const dwih = e("(h_err .* h_dAdZ) * input'", {
            h_err,
            h_dAdZ,
            input,
        });

        this.cache.dwih = dwih;
        this.cache.dwho = dwho;
        this.cache.loss.push(sum(sqrt(dEdA)));
    };

    update = () => { /*...*/ };
    predict = (input) => { /*...*/ };
    train = (input, target) => { /*...*/ };
}