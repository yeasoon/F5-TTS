#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <fstream>
#include <cstdint>
#include <string>
using namespace std;
#include <stdexcept>

using Complex = std::complex<double>;
const double PI = 3.14159265358979323846f;

// In-place radix-2 FFT
void bitReverse(std::vector<Complex>& a) {
    size_t n = a.size();
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

void fft(std::vector<Complex>& a) {
    size_t n = a.size();
    if ((n & (n - 1)) != 0) throw std::runtime_error("FFT size must be power of 2");
    bitReverse(a);
    for (size_t len = 2; len <= n; len <<= 1) {
        double angle = -2.0f * PI / len;
        Complex wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0f, 0.0f);
            for (size_t j = 0; j < len / 2; ++j) {
                Complex u = a[i+j];
                Complex v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Generate periodic Hann window (like PyTorch default)
std::vector<double> hannWindow(size_t win_size) {
    std::vector<double> w(win_size);
    for (size_t i = 0; i < win_size; ++i)
        w[i] = 0.5f - 0.5f * std::cos(2.0f * PI * i / win_size); // periodic=True
    return w;
}

std::vector<double> reflect_pad_1d(const std::vector<double>& y, int pad_left, int pad_right) {
    if (y.empty()) return {};

    std::vector<double> padded;
    padded.reserve(pad_left + y.size() + pad_right);

    // ---- Left reflect ----
    for (int i = 0; i < pad_left; ++i) {
        int idx = pad_left - i; // reflect excluding edge
        if (idx >= (int)y.size()) idx = y.size() - 1;
        padded.push_back(y[idx]);
    }

    // ---- Original ----
    padded.insert(padded.end(), y.begin(), y.end());

    // ---- Right reflect ----
    for (int i = 0; i < pad_right; ++i) {
        int idx = y.size() - 2 - i; // reflect excluding last
        if (idx < 0) idx = 0;
        padded.push_back(y[idx]);
    }

    return padded;
}
// Spectrogram similar to PyTorch
std::vector<double> spectrogram_torch(
    const std::vector<double>& y,
    size_t n_fft,
    size_t hop_size,
    size_t win_size,
    bool center = true
) {
    size_t len = y.size();
    auto pad = (n_fft - hop_size)/2;
    auto signal = reflect_pad_1d(y, pad, pad);
    std::vector<double> window = hannWindow(win_size);
    size_t num_frames = (signal.size() - win_size) / hop_size+1;
    std::vector<double> spectrogram(num_frames*(n_fft/2 + 1), 0.0f);

    for (size_t frame = 0; frame < num_frames; ++frame) {
        size_t start = frame * hop_size;
        std::vector<Complex> buf(n_fft, 0.0f);

        // Apply window and copy to FFT buffer
        for (size_t i = 0; i < win_size; ++i)
            buf[i] = signal[start + i] * window[i];

        fft(buf);

        // Compute onesided magnitude with epsilon
        // std::vector<double> mag(n_fft/2 + 1);
        for (size_t i = 0; i <= n_fft/2; ++i)
            spectrogram[frame*(n_fft/2 + 1)+i] = std::sqrt(std::norm(buf[i]) + 1e-6f);

        // spectrogram.push_back(mag);
    }

    return spectrogram;
}

struct WavData {
    int sampleRate = 0;
    int numChannels = 0;
    int bitsPerSample = 0;
    std::vector<double> samples; // normalized [-1, 1]
};



// Read little-endian integer from file
template<typename T>
T readLE(std::ifstream &f) {
    T val;
    f.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}

WavData loadWav(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    // Check RIFF header
    char riff[4];
    f.read(riff, 4);
    if (std::string(riff, 4) != "RIFF")
        throw std::runtime_error("Not a RIFF file");

    f.seekg(8); // skip size + WAVE

    WavData wav;
    uint16_t audioFormat = 0;
    bool fmtRead = false;
    bool dataRead = false;
    auto  file_len = readLE<uint16_t>(f);
    auto  file_fmt= readLE<uint16_t>(f);

    while (f && !dataRead) {
        char chunkId[4];
        if (!f.read(chunkId, 4)) break;

        uint32_t chunkSize = readLE<uint32_t>(f);
        std::string id(chunkId, 4);
        std::streampos chunkStart = f.tellg();

        if (id == "fmt ") {
            audioFormat = readLE<uint16_t>(f);
            wav.numChannels = readLE<uint16_t>(f);
            wav.sampleRate = readLE<uint32_t>(f);
            uint32_t byteRate = readLE<uint32_t>(f);
            uint16_t blockAlign = readLE<uint16_t>(f);
            wav.bitsPerSample = readLE<uint16_t>(f);
            std::cout << "  Channels: " << wav.numChannels << "\n";
            std::cout << "  Sample rate: " << wav.sampleRate << "\n";
            std::cout << "  Bits: " << wav.bitsPerSample << "\n";


            if (audioFormat != 1)
                throw std::runtime_error("Only PCM WAV supported");

            fmtRead = true;

            // Skip any extra bytes in fmt chunk
            std::streamoff remaining = chunkSize - 16;
            if (remaining > 0) {
                f.seekg(remaining, std::ios::cur);
            }

        } else if (id == "data") {
            if (!fmtRead)
                throw std::runtime_error("Found 'data' before 'fmt ' chunk");

            size_t bytesPerSample = wav.bitsPerSample / 8;
            size_t numValues = chunkSize / bytesPerSample;

            wav.samples.resize(numValues);

            if (wav.bitsPerSample == 16) {
                for (size_t i = 0; i < numValues; ++i) {
                    int16_t s = readLE<int16_t>(f);
                    wav.samples[i] = s / 32768.0f;
                }
            } else if (wav.bitsPerSample == 8) {
                for (size_t i = 0; i < numValues; ++i) {
                    uint8_t s = readLE<uint8_t>(f);
                    wav.samples[i] = (s - 128) / 128.0f;
                }
            } else if (wav.bitsPerSample == 32) {
                for (size_t i = 0; i < numValues; ++i) {
                    int32_t s = readLE<int32_t>(f);
                    wav.samples[i] = s / 2147483648.0f;
                }
            } else {
                throw std::runtime_error("Unsupported bits per sample");
            }

            dataRead = true;
        }

        // Skip unknown chunks (LIST, cue, fact, etc.) with padding
        std::streamoff skip = chunkSize;
        if (skip % 2 != 0) skip++; // pad to even
        f.seekg(chunkStart + skip);
    }

    if (!dataRead)
        throw std::runtime_error("No 'data' chunk found");

    f.close();
    return wav;
}

std::vector<float> load_bin_float32(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Failed to open file: " + path);

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size % sizeof(float) != 0)
        throw std::runtime_error("File size not aligned to float32 samples.");

    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
        throw std::runtime_error("Failed to read file: " + path);

    return buffer;
}

std::vector<double> load_bin_double(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Failed to open file: " + path);

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size % sizeof(double) != 0)
        throw std::runtime_error("File size not aligned to float32 samples.");

    std::vector<double> buffer(size / sizeof(double));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
        throw std::runtime_error("Failed to read file: " + path);

    return buffer;
}

class LayerNorm {
public:
    LayerNorm(const std::string& gamma_init,
              const std::string& beta_init,
              float epsilon = 1e-5f)
        :  eps(epsilon) {
        gamma = load_bin_float32(gamma_init);
        beta = load_bin_float32(beta_init);
        hidden_dim = gamma.size();
    }

    // Input: 3D tensor [B][T][H]
    void forward(std::vector<double>& x,size_t B, size_t T,size_t H) const {
        if (B == 0) return;
        if (T == 0) return;
        for (size_t b = 0; b < B; ++b) {
            for (size_t t = 0; t < T; ++t) {
                // Compute mean
                double mean = 0.0f;
                for (size_t h = 0; h < H; ++h){
                    auto v=x[b*T*H+t*H+h];
                    mean +=v;
                }
                mean /= hidden_dim;

                // Compute variance
                double var = 0.0f;
                for (size_t h = 0; h < H; ++h){
                    auto v=x[b*T*H+t*H+h];
                    var += (v - mean) * (v - mean);
                }
                var /= hidden_dim;

                double inv_std = 1.0f / std::sqrt(var + eps);

                // Normalize and apply gamma/beta
                for (size_t h = 0; h < hidden_dim; ++h)
                    x[b*T*H+t*H+h] = (x[b*T*H+t*H+h] - mean) * inv_std * double(gamma[h]) + double(beta[h]);
            }
        }
    }

    size_t dim() const { return hidden_dim; }

private:
    std::vector<float> gamma;
    std::vector<float> beta;
    float eps;
    size_t hidden_dim;
};

class Conv2dWeightNorm {
public:
    Conv2dWeightNorm(const std::string& v_init, const std::string& g_init,  const std::string& b_init, int in_c, int out_c, int k, int stride=1, int pad=0, int relu=0)
        : in_channels(in_c), out_channels(out_c),
          kernel_size(k), stride(stride), padding(pad), relu(relu)
    {
        // Initialize v 
        v = load_bin_float32(v_init);
        // Initialize g 
        g = load_bin_float32(g_init);
        // Initialize bias to zeros
        bias  =  load_bin_float32(b_init);
    }

    // Forward pass: input shape [B, C_in, H, W]
    std::vector<double> forward(const std::vector<double>& input, int B,  int H_in, int W_in)
    {
        // Reconstruct normalized weight
        auto w = get_weight_normalized();  // make sure w is vector<double>
        int H_out = (H_in + 2*padding - kernel_size) / stride + 1;
        int W_out = (W_in + 2*padding - kernel_size) / stride + 1;
        std::vector<double> output(B * out_channels * H_out * W_out, 0.0);
        std::cout <<H_out<<" "<< W_out << std::endl;
        // Convolution
        for (int b = 0; b < B; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        double sum = static_cast<double>(bias[oc]);
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int ih = oh*stride + kh - padding;
                                    int iw = ow*stride + kw - padding;
                                    double val = 0.0;
                                    if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in)
                                        val = input[((b*in_channels + ic)*H_in + ih)*W_in + iw];
                                    else
                                        // std::cout <<ih<<" "<< iw << std::endl;
                                        continue;
                                    int weight_idx = ((oc*in_channels + ic)*kernel_size + kh)*kernel_size + kw;
                                    sum += val * w[weight_idx];
                                }
                            }
                        }
                        int out_idx = ((b*out_channels + oc)*H_out + oh)*W_out + ow;
                        if (relu==1)
                            sum = std::max(0.0, sum);
                        output[out_idx] = sum;  // keep as double
                    }
                }
            }
        }

        return output;
    }
    std::vector<int> get_out_shape(int B,  int H_in, int W_in){
        std::vector<int> shape(4);
        shape[0]=B;
        shape[1]=out_channels;
        shape[2] = (H_in + 2*padding - kernel_size) / stride + 1;
        shape[3] = (W_in + 2*padding - kernel_size) / stride + 1;
        return shape;
    }

private:
    int in_channels, out_channels, kernel_size, stride, padding, relu;
    std::vector<float> v;    // direction
    std::vector<float> g;    // scale per output channel
    std::vector<float> bias; // bias

    double compute_norm(int oc) const {
        double sum_sq = 0.0;
        for (int ic = 0; ic < in_channels; ++ic)
            for (int kh = 0; kh < kernel_size; ++kh)
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int idx = ((oc * in_channels + ic) * kernel_size * kernel_size) + (kh * kernel_size) + kw;
                    sum_sq += v[idx] * v[idx]; // v should be double
                }
        return std::sqrt(sum_sq);
    }

    std::vector<double> get_weight_normalized() const {
        std::vector<double> w(v.size());
        for (int oc = 0; oc < out_channels; ++oc) {
            double norm = compute_norm(oc);
            double scale = g[oc] / (norm + 1e-6);  // double literal
            
            for (int ic = 0; ic < in_channels; ++ic)
                for (int kh = 0; kh < kernel_size; ++kh)
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int idx = ((oc * in_channels + ic) * kernel_size * kernel_size) + (kh * kernel_size) + kw;
                        w[idx] = v[idx] * scale;
                        // if (oc==0){
                        //     std::cout << w[idx] <<std::endl;
                        // }
                    }
        }
        return w;
    }
};


#include <algorithm>

class GRU {
public:
    // C: input channels, F: features per channel, hidden_size: GRU hidden dim
    GRU(const std::string& W_ih_init,
        const std::string& W_hh_init,
        const std::string& b_ih_init,
        const std::string& b_hh_init,
        int C, int F, int hidden_size)
        : C(C), F(F), hidden_size(hidden_size) 
    {
        input_size = C * F; // flatten channels * features
        W_ih = load_bin_double(W_ih_init); // shape [3*hidden_size, input_size]
        W_hh = load_bin_double(W_hh_init); // shape [3*hidden_size, hidden_size]
        b_ih = load_bin_double(b_ih_init); // shape [3*hidden_size]
        b_hh = load_bin_double(b_hh_init); // shape [3*hidden_size]
    }

    std::vector<double> forward(const std::vector<double>& input, int B, int T) {
        std::vector<double> h(B * hidden_size, 0.0);
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                double* h_b = &h[b * hidden_size];

                std::vector<double> r(hidden_size);
                std::vector<double> z(hidden_size);
                std::vector<double> n(hidden_size);

                for (int i = 0; i < hidden_size; ++i) {
                    double r_i  = 0.0; 
                    double z_i = 0.0;
                    double n_i = 0.0;

                    const double* x_t = &input[(b * T + t) * input_size];
                    for (int j = 0; j < input_size; ++j) {
                        r_i += W_ih[i * input_size + j] * x_t[j];
                        z_i += W_ih[(hidden_size + i) * input_size + j] * x_t[j];
                        n_i += W_ih[(2 * hidden_size + i) * input_size + j] * x_t[j];
                    }

                    r_i += b_ih[i];
                    z_i += b_ih[hidden_size + i];
                    n_i += b_ih[2 * hidden_size + i];
                    double n_s = 0.0;
                    for (int j = 0; j < hidden_size; ++j) {
                        double h_prev = h_b[j];
                        r_i += W_hh[i * hidden_size + j] * h_prev;
                        z_i += W_hh[(hidden_size + i) * hidden_size + j] * h_prev;
                        n_s += W_hh[(2 * hidden_size + i) * hidden_size + j] * h_prev;
                    }
                    r_i += b_hh[i];
                    z_i += b_hh[hidden_size + i];
                    // std::cout << r_i << " "<<z_i <<std::endl;
                    r[i] = sigmoid(r_i);
                    z[i] = sigmoid(z_i);
                    r[i] = std::min(std::max(r[i], 1e-5), 1 - 1e-5);
                    z[i] = std::min(std::max(z[i], 1e-5), 1 - 1e-5);

                    // z[i] = z_i;
                    // std::cout << r[i] << " "<<z[i] <<std::endl;
                    n_i+=r[i]*(n_s+b_hh[2 * hidden_size + i]);
                    // n[i] = stable_tanh(n_i);
                    n[i] = std::tanh(std::clamp(n_i, -40.0, 40.0));
                    // n[i] = n_i;
                }

                // Update hidden
                for (int i = 0; i < hidden_size; ++i) {
                    h_b[i] = (1 - z[i]) * n[i] + z[i] * h_b[i];
                    // h_b[i] = n[i];
                    // h_b[i] = z[i] + r[i];
                }
            }
        }

        return h;
    }

private:
    int C, F, input_size, hidden_size;
    std::vector<double> W_ih, W_hh, b_ih, b_hh;

    inline double sigmoid(double v) const {
        // double v = std::clamp(x, -40.0, 40.0);
        if (v >= 0)
            return 1.0 / (1.0 + std::exp(-v));
        else {
            double exp_v = std::exp(v);
            return exp_v / (1.0 + exp_v);
        }
    }
    inline double stable_tanh(double x) {
    // if (x >= 20) return 1.0;
    // if (x <= -20) return -1.0;
    double e2x = std::exp(2 * x);
    return (e2x - 1) / (e2x + 1);
}
};

class Linear{
public:
    Linear(const std::string& w_init, const std::string& b_init,  int in_c, int out_c)
        : in_channels(in_c), out_channels(out_c)
    {
        // Initialize v 
        W = load_bin_float32(w_init);
        // Initialize g 
        b = load_bin_float32(b_init);
    }
    std::vector<double> forward(const std::vector<double>& x, int B)
    {
        std::vector<double> y(B * out_channels, 0.0);

        for (int n = 0; n < B; ++n) {
            for (int o = 0; o < out_channels; ++o) {
                double sum = b[o];
                for (int i = 0; i < in_channels; ++i) {
                    sum += x[n * in_channels + i] * W[o * in_channels + i];
                }
                y[n * out_channels + o] = sum;
            }
        }
        return y;
    }
private:
    int in_channels, out_channels;
    std::vector<float> W;    // direction
    std::vector<float> b;    // scale per output channel
};
int main() {
    // Generate 1 kHz sine wave
    // int sr = 22050;
    // double freq = 1000.0;
    // double duration = 1.0;
    // int samples = int(sr * duration);
    // vector<double> y(samples);
    // for (int i = 0; i < samples; ++i)
    //     y[i] = sin(2 * M_PI * freq * i / sr);

    try {
        WavData wav = loadWav("/data/tts/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en_pcm.wav");
        std::cout << "Loaded WAV:\n";
        std::cout << "  Channels: " << wav.numChannels << "\n";
        std::cout << "  Sample rate: " << wav.sampleRate << "\n";
        std::cout << "  Bits: " << wav.bitsPerSample << "\n";
        std::cout << "  Samples: " << wav.samples.size() << "\n";
        std::cout << "  First 10 samples: ";
        for (int i = 0; i < 10 && i < wav.samples.size(); ++i)
            std::cout << wav.samples[i] << " ";
        std::cout << "\n";
        auto spec = spectrogram_torch(wav.samples, 1024, 256, 1024);
        // cout << "Frames: " << spec.size() << ", Freq bins: " << spec[0].size() << endl;
        // for (int i = 0; i < 1; ++i){
        //     for (int j = 0; j < 10; ++j)
        //         cout << spec[i][j] << " ";
        //     cout << endl;
        // }
        // cout << endl;
        LayerNorm layernorm("/data/tts/OpenVoice/models_param/ref_enc.layernorm.weight.bin", 
                            "/data/tts/OpenVoice/models_param/ref_enc.layernorm.bias.bin");
        layernorm.forward(spec, 1, 459, 513);
        std::vector<int> filter={1,32, 32, 64, 64, 128, 128};
        int l=0;
        Conv2dWeightNorm conv_0("/data/tts/OpenVoice/models_param/ref_enc.conv_0.weight_v.bin", 
        "/data/tts/OpenVoice/models_param/ref_enc.conv_0.weight_g.bin",
        "/data/tts/OpenVoice/models_param/ref_enc.conv_0.bias.bin", filter[l], filter[l+1],3,2,1,1);
        auto out=conv_0.forward(spec, 1, 459,513);
        auto shape=conv_0.get_out_shape(1, 459,513);
        for (int i = 0; i < 1; ++i){
            for (int j = 0; j < 10; ++j)
                cout << out[i*shape[3]+j] << " ";
            cout << endl;
        }
        cout << endl;
        l=1;
        Conv2dWeightNorm conv_1("/data/tts/OpenVoice/models_param/ref_enc.conv_1.weight_v.bin", 
        "/data/tts/OpenVoice/models_param/ref_enc.conv_1.weight_g.bin",
        "/data/tts/OpenVoice/models_param/ref_enc.conv_1.bias.bin", filter[l], filter[l+1],3,2,1,1);
        auto out1=conv_1.forward(out, shape[0], shape[2], shape[3]);
        auto shape1=conv_1.get_out_shape(shape[0], shape[2], shape[3]);
        for (int i = 0; i < 1; ++i){
            for (int j = 0; j < 10; ++j)
                cout << out1[i*shape1[3]+j] << " ";
            cout << endl;
        }
        cout << endl;
        l=2;
        Conv2dWeightNorm conv_2("/data/tts/OpenVoice/models_param/ref_enc.conv_2.weight_v.bin", 
        "/data/tts/OpenVoice/models_param/ref_enc.conv_2.weight_g.bin",
        "/data/tts/OpenVoice/models_param/ref_enc.conv_2.bias.bin", filter[l], filter[l+1],3,2,1,1);
        auto out2=conv_2.forward(out1, shape1[0], shape1[2], shape1[3]);
        auto shape2=conv_2.get_out_shape(shape1[0], shape1[2], shape1[3]);
        for (int i = 0; i < 1; ++i){
            for (int j = 0; j < 10; ++j)
                cout << out2[i*shape2[3]+j] << " ";
            cout << endl;
        }
        cout << endl;

        l=3;
        Conv2dWeightNorm conv_3("/data/tts/OpenVoice/models_param/ref_enc.conv_3.weight_v.bin", 
        "/data/tts/OpenVoice/models_param/ref_enc.conv_3.weight_g.bin",
        "/data/tts/OpenVoice/models_param/ref_enc.conv_3.bias.bin", filter[l], filter[l+1],3,2,1,1);
        auto   out3=conv_3.forward(out2, shape2[0], shape2[2], shape2[3]);
        auto shape3=conv_3.get_out_shape(shape2[0], shape2[2], shape2[3]);
        for (int i = 0; i < 1; ++i){
            for (int j = 0; j < 10; ++j)
                cout << out3[i*shape3[3]+j] << " ";
            cout << endl;
        }
        cout << endl;

        l=4;
        Conv2dWeightNorm conv_4(
        "/data/tts/OpenVoice/models_param/ref_enc.conv_4.weight_v.bin", 
        "/data/tts/OpenVoice/models_param/ref_enc.conv_4.weight_g.bin",
        "/data/tts/OpenVoice/models_param/ref_enc.conv_4.bias.bin", filter[l], filter[l+1],3,2,1,1);
        auto   out4=conv_4.forward(out3, shape3[0], shape3[2], shape3[3]);
        auto shape4=conv_4.get_out_shape(shape3[0], shape3[2], shape3[3]);
        for (int i = 0; i < 1; ++i){
            for (int j = 0; j < 10; ++j)
                cout << out4[i*shape4[3]+j] << " ";
            cout << endl;
        }
        cout << endl;

        l=5;
        Conv2dWeightNorm conv_5(
        "/data/tts/OpenVoice/models_param/ref_enc.conv_5.weight_v.bin", 
        "/data/tts/OpenVoice/models_param/ref_enc.conv_5.weight_g.bin",
        "/data/tts/OpenVoice/models_param/ref_enc.conv_5.bias.bin", filter[l], filter[l+1],3,2,1,1);
        auto   out5=conv_5.forward(out4, shape4[0], shape4[2], shape4[3]);
        auto shape5=conv_5.get_out_shape(shape4[0], shape4[2], shape4[3]);
        for (int i = 0; i < 1; ++i){
            for (int j = 0; j < 10; ++j)
                cout << out5[i*shape5[3]+j] << " ";
            cout << endl;
        }
        cout << endl;
        GRU gru(
        "/data/tts/OpenVoice/models_param/ref_enc.gru.weight_ih_l0.bin", 
        "/data/tts/OpenVoice/models_param/ref_enc.gru.weight_hh_l0.bin",
        "/data/tts/OpenVoice/models_param/ref_enc.gru.bias_ih_l0.bin", 
        "/data/tts/OpenVoice/models_param/ref_enc.gru.bias_hh_l0.bin", 
        shape5[1], shape5[3], 128);
        std::vector<double> tmp(out5.size(), 0.0);
        for (int b = 0; b <  shape5[0]; ++b){
            for (int c = 0; c <  shape5[1]; ++c){
                for (int t = 0; t <  shape5[2]; ++t){
                    for (int f = 0; f <  shape5[3]; ++f){
                        int idx0=((b*shape5[1]+c)*shape5[2]+t)*shape5[3]+f;
                        int idx1=((b*shape5[2]+t)*shape5[1]+c)*shape5[3]+f;
                        tmp[idx1]=out5[idx0];
                    }
                }
            }
            
        }
        for (int j = 0; j < 10; ++j)
            cout << tmp[j*128*9] << " ";
        cout << endl;
        // 1152,128);
        std::cout <<shape5[0]<< " "<< shape5[1] << " "<< shape5[2]<< " "<<shape5[3]<< " "<<std::endl;
        // auto tmp1=load_bin_double("/data/tts/OpenVoice/models_param/dbg.bin");
        // for (auto idx=0;idx < tmp1.size();++idx){
        //     cout << tmp[idx]-tmp1[idx] << " ";
        // }
        // cout << endl;
        auto  out6=gru.forward(tmp, shape5[0], shape5[2]);
        for (int i = 0; i < 1; ++i){
            for (int j = 0; j < 10; ++j)
                cout << out6[i*shape5[3]+j] << " ";
            cout << endl;
        }
        cout << endl;
        // auto tmp1=load_bin_double("/data/tts/OpenVoice/models_param/dbg.bin");
        Linear proj(
        "/data/tts/OpenVoice/models_param/ref_enc.proj.weight.bin", 
        "/data/tts/OpenVoice/models_param/ref_enc.proj.bias.bin", 
        128, 256);
        auto  out7=proj.forward(out6, shape5[0]);
        for (int j = 0; j < 10; ++j)
            cout << out7[j] << " ";
        cout << endl;
        
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    


    return 0;
}