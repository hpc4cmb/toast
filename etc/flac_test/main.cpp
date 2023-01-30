
#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>

#include <random>
#include <vector>
#include <unordered_map>
#include <map>
#include <limits>

#include <FLAC/stream_encoder.h>
#include <FLAC/stream_decoder.h>


void fake_data(std::vector <int32_t> & data) {
    std::random_device dev;
    std::mt19937 rng(dev());

    int32_t max = std::numeric_limits<int32_t>::max() / 2;
    int32_t min = -max;
    std::uniform_int_distribution<std::mt19937::result_type> dist(min, max);

    for (auto & p : data) {
        p = dist(rng);
    }
    return;
}



typedef struct {
    std::vector <uint8_t> * compressed;
} enc_write_callback_data;


FLAC__StreamEncoderWriteStatus enc_write_callback(
    const FLAC__StreamEncoder *encoder, 
    const FLAC__byte buffer[], 
    size_t bytes, 
    uint32_t samples, 
    uint32_t current_frame, 
    void *client_data
) {
    enc_write_callback_data * data = (enc_write_callback_data *)client_data;
    std::cerr << "Encode frame " << current_frame << " got " << bytes << " bytes for " << samples << " samples" << std::endl;
    data->compressed->insert(
        data->compressed->end(), 
        buffer, 
        buffer + bytes
    );
    return FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
}


void encode_flac(
    std::vector <int32_t> const & data, 
    std::vector <uint8_t> & bytes,
    std::vector <int64_t> & offsets,
    uint32_t level, 
    int64_t stride
) {
    // If stride is specified, check consistency.
    int64_t n_sub;
    if (stride > 0) {
        if (data.size() % stride != 0) {
            std::cerr << "Stride " << stride << " does not evenly divide into " << data.size() << std::endl;
            return;
        }
        n_sub = (int64_t)(data.size() / stride);
    } else {
        n_sub = 1;
        stride = data.size();
    }
    offsets.resize(n_sub);
    bytes.clear();

    enc_write_callback_data write_callback_data;
    write_callback_data.compressed = &bytes;

    // settings

    bool success;

    FLAC__StreamEncoderInitStatus status;

    FLAC__StreamEncoder * encoder;

    for (int64_t sub = 0; sub < n_sub; ++sub) {
        offsets[sub] = bytes.size();

        std::cerr << "Encoding " << stride << " samples at byte offset " << offsets[sub] << " starting at data element " << (sub * stride) << std::endl;

        encoder = FLAC__stream_encoder_new();

        success = FLAC__stream_encoder_set_compression_level(encoder, level);
        if (! success) {
            std::cerr << "Failed to set compression level" << std::endl;
            return;
        }

        success = FLAC__stream_encoder_set_blocksize(encoder, 0);
        if (! success) {
            std::cerr << "Failed to set blocksize" << std::endl;
            return;
        }

        success = FLAC__stream_encoder_set_channels(encoder, 1);
        if (! success) {
            std::cerr << "Failed to set channels" << std::endl;
            return;
        }

        success = FLAC__stream_encoder_set_bits_per_sample(encoder, 32);
        if (! success) {
            std::cerr << "Failed to set bits per sample" << std::endl;
            return;
        }

        status = FLAC__stream_encoder_init_stream(
            encoder, 
            enc_write_callback, 
            NULL, 
            NULL, 
            NULL, 
            (void *)&write_callback_data
        );
        if (status != FLAC__STREAM_ENCODER_INIT_STATUS_OK) {
            std::cerr << "Failed to init encoder, status = " << status << std::endl;
            return;
        }

        success = FLAC__stream_encoder_process_interleaved(
            encoder,
            &(data[sub * stride]),
            stride
        );
        if (! success) {
            std::cerr << "  Failed" << std::endl;
            FLAC__StreamEncoderState state = FLAC__stream_encoder_get_state(encoder);
            std::cerr << "  state was " << state << std::endl;
            break;
        }
        success = FLAC__stream_encoder_finish(encoder);
        if (! success) {
            std::cerr << "Failed encode finish" << std::endl;
        }

        FLAC__stream_encoder_delete(encoder);
    }

    return;
}

// ------------- decoding -------------------

typedef struct {
    std::vector <uint8_t> const * input;
    size_t in_offset;
    size_t in_end;
    std::vector <int32_t> * output;
} dec_callback_data;


FLAC__StreamDecoderReadStatus dec_read_callback(
    const FLAC__StreamDecoder * decoder, 
    FLAC__byte buffer[], 
    size_t * bytes, 
    void * client_data
) {
    dec_callback_data * callback_data = (dec_callback_data*)client_data;
    std::vector <uint8_t> const * input = callback_data->input;
    size_t offset = callback_data->in_offset;
    size_t remaining = callback_data->in_end - offset;
    std::cerr << "Decode read:  " << remaining << " bytes remaining" << std::endl;

    // The bytes requested by the decoder
    size_t n_buffer = (*bytes);

    if (remaining == 0) {
        // No data left
        (*bytes) = 0;
        std::cerr << "Decode read:  0 bytes remaining, END_OF_STREAM" << std::endl;
        return FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;
    } else {
        // We have some data left
        if (n_buffer == 0) {
            // ... but there is no place to put it!
            std::cerr << "Decode read:  0 bytes in buffer, ABORT" << std::endl;
            return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
        } else {
            if (remaining > n_buffer) {
                // Only copy in what there is space for
                std::cerr << "Decode read:  putting " << n_buffer << " bytes at offset " << offset << " into buffer, CONTINUE" << std::endl;
                for (size_t i = 0; i < n_buffer; ++i) {
                    buffer[i] = (*input)[offset + i];
                }
                callback_data->in_offset += n_buffer;
                return FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
            } else {
                // Copy in the rest of the buffer and reset the number of bytes
                std::cerr << "Decode read:  putting remainder of " << remaining << " bytes at offset " << offset << " into buffer, CONTINUE" << std::endl;
                for (size_t i = 0; i < remaining; ++i) {
                    buffer[i] = (*input)[offset + i];
                }
                callback_data->in_offset += remaining;
                (*bytes) = remaining;
                return FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
            }
        }
    }
    // Should never get here...
    return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
}


FLAC__StreamDecoderWriteStatus dec_write_callback(
    const FLAC__StreamDecoder * decoder, 
    const FLAC__Frame * frame, 
    const FLAC__int32 * const buffer[], 
    void * client_data
) {
    dec_callback_data * data = (dec_callback_data *)client_data;
    size_t offset = data->output->size();
    uint32_t blocksize = frame->header.blocksize;
    data->output->resize(offset + blocksize);
    for (size_t i = 0; i < blocksize; ++i) {
        (*data->output)[offset + i] = buffer[0][i];
    }
    return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
}


void dec_err_callback(
    const FLAC__StreamDecoder * decoder,
    FLAC__StreamDecoderErrorStatus status,
    void *client_data
) {
    dec_callback_data * data = (dec_callback_data *)client_data;

    return;
}


void decode_flac(
    std::vector <uint8_t> const & bytes,
    std::vector <int64_t> const & offsets,
    std::vector <int32_t> & data
) {
    dec_callback_data callback_data;
    callback_data.input = &bytes;
    callback_data.output = &data;

    FLAC__StreamDecoder * decoder;


    bool success;

    FLAC__StreamDecoderInitStatus status;
    

    size_t n_sub = offsets.size();

    for (size_t sub = 0; sub < n_sub; ++sub) {
        callback_data.in_offset = offsets[sub];
        if (sub == n_sub - 1) {
            callback_data.in_end = bytes.size();
        } else {
            callback_data.in_end = offsets[sub + 1];
        }
        std::cerr << "Decoding chunk " << sub << " at byte offset " << callback_data.in_offset << " with " << (callback_data.in_end - callback_data.in_offset) << " bytes" << std::endl;
        
        decoder = FLAC__stream_decoder_new();

        status = FLAC__stream_decoder_init_stream(
            decoder, 
            dec_read_callback, 
            NULL, 
            NULL, 
            NULL, 
            NULL,
            dec_write_callback,
            NULL,
            dec_err_callback,
            (void *)&callback_data
        );
        if (status != FLAC__STREAM_DECODER_INIT_STATUS_OK) {
            std::cerr << "Failed to init decoder, status = " << status << std::endl;
            return;
        }
        
        success = FLAC__stream_decoder_process_until_end_of_stream(decoder);
        if (! success) {
            std::cerr << "  Failed" << std::endl;
            break;
        }

        success = FLAC__stream_decoder_finish(decoder);
        if (! success) {
            std::cerr << "Failed decode finish" << std::endl;
        }

        FLAC__stream_decoder_delete(decoder);
    }
    

    return;
}


int main(int argc, char * argv[]) {

    int64_t n_det = 10;
    int64_t n_samp = 100000;

    std::vector <int32_t> data(n_samp * n_det);
    std::vector <uint8_t> compressed;
    std::vector <int64_t> offsets;
    std::vector <int32_t> output;

    fake_data(data);

    encode_flac(data, compressed, offsets, 5, n_samp);

    std::cout << "Input was " << n_samp * n_det * 8 << " bytes, compress to " << compressed.size() << " bytes" << std::endl;

    decode_flac(compressed, offsets, output);

    for (int64_t i = 0; i < n_det; ++i) {
        for (int64_t j = 0; j < n_samp; ++j) {
            if (data[i * n_samp + j] != output[i * n_samp + j]) {
                std::cerr << "Det " << i << ", sample " << j << ": " << output[i * n_samp + j] << " != " << data[i * n_samp + j] << std::endl;
            }
        }
    }
    
    
    return 0;
}

