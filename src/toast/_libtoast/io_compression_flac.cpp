
// Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>


#ifdef HAVE_FLAC
# include <FLAC/stream_encoder.h>
# include <FLAC/stream_decoder.h>


typedef struct {
    toast::AlignedU8 * compressed;
} enc_write_callback_data;


FLAC__StreamEncoderWriteStatus enc_write_callback(
    const FLAC__StreamEncoder * encoder,
    const FLAC__byte buffer[],
    size_t bytes,
    uint32_t samples,
    uint32_t current_frame,
    void * client_data
) {
    enc_write_callback_data * data = (enc_write_callback_data *)client_data;
    data->compressed->insert(
        data->compressed->end(),
        buffer,
        buffer + bytes
    );
    return FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
}

void encode_flac(
    int32_t * const data,
    size_t n_data,
    toast::AlignedU8 & bytes,
    toast::AlignedI64 & offsets,
    uint32_t level,
    int64_t stride
) {
    // If stride is specified, check consistency.
    int64_t n_sub;
    if (stride > 0) {
        if (n_data % stride != 0) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Stride " << stride << " does not evenly divide into " << n_data;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
        n_sub = (int64_t)(n_data / stride);
    } else {
        n_sub = 1;
        stride = n_data;
    }
    offsets.resize(n_sub);
    bytes.clear();

    enc_write_callback_data write_callback_data;
    write_callback_data.compressed = &bytes;

    bool success;
    FLAC__StreamEncoderInitStatus status;
    FLAC__StreamEncoder * encoder;

    for (int64_t sub = 0; sub < n_sub; ++sub) {
        offsets[sub] = bytes.size();

        // std::cerr << "Encoding " << stride << " samples at byte offset " <<
        // offsets[sub] << " starting at data element " << (sub * stride) << std::endl;

        encoder = FLAC__stream_encoder_new();

        success = FLAC__stream_encoder_set_compression_level(encoder, level);
        if (!success) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Failed to set compression level to " << level;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }

        success = FLAC__stream_encoder_set_blocksize(encoder, 0);
        if (!success) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Failed to set encoder blocksize to " << 0;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }

        success = FLAC__stream_encoder_set_channels(encoder, 1);
        if (!success) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Failed to set encoder channels to " << 1;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }

        success = FLAC__stream_encoder_set_bits_per_sample(encoder, 32);
        if (!success) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Failed to set encoder bits per sample to " << 32;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
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
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Failed to initialize stream encoder, status = " << status;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }

        success = FLAC__stream_encoder_process_interleaved(
            encoder,
            &(data[sub * stride]),
            stride
        );
        if (!success) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Failed on encoder_process_interleaved for chunk " << sub;
            o << ", elements " << sub * stride << " - " << (sub + 1) * stride;
            o << ", at byte offset " << offsets[sub];
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
        success = FLAC__stream_encoder_finish(encoder);
        if (!success) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Failed on encoder_finish";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }

        FLAC__stream_encoder_delete(encoder);
    }

    return;
}

typedef struct {
    uint8_t const * input;
    size_t in_nbytes;
    size_t in_offset;
    size_t in_end;
    toast::AlignedI32 * output;
} dec_callback_data;


FLAC__StreamDecoderReadStatus dec_read_callback(
    const FLAC__StreamDecoder * decoder,
    FLAC__byte buffer[],
    size_t * bytes,
    void * client_data
) {
    dec_callback_data * callback_data = (dec_callback_data *)client_data;
    uint8_t const * input = callback_data->input;
    size_t offset = callback_data->in_offset;
    size_t remaining = callback_data->in_end - offset;

    // std::cerr << "Decode read:  " << remaining << " bytes remaining" << std::endl;

    // The bytes requested by the decoder
    size_t n_buffer = (*bytes);

    if (remaining == 0) {
        // No data left
        (*bytes) = 0;

        // std::cerr << "Decode read:  0 bytes remaining, END_OF_STREAM" << std::endl;
        return FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;
    } else {
        // We have some data left
        if (n_buffer == 0) {
            // ... but there is no place to put it!
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Stream decoder gave us zero length buffer, but we have ";
            o << remaining << " bytes left";
            log.error(o.str().c_str());

            // std::cerr << "Decode read:  0 bytes in buffer, ABORT" << std::endl;
            return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
        } else {
            if (remaining > n_buffer) {
                // Only copy in what there is space for
                // std::cerr << "Decode read:  putting " << n_buffer << " bytes at
                // offset " << offset << " into buffer, CONTINUE" << std::endl;
                for (size_t i = 0; i < n_buffer; ++i) {
                    buffer[i] = input[offset + i];
                }
                callback_data->in_offset += n_buffer;
                return FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
            } else {
                // Copy in the rest of the buffer and reset the number of bytes
                // std::cerr << "Decode read:  putting remainder of " << remaining << "
                // bytes at offset " << offset << " into buffer, CONTINUE" << std::endl;
                for (size_t i = 0; i < remaining; ++i) {
                    buffer[i] = input[offset + i];
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
    void * client_data
) {
    dec_callback_data * data = (dec_callback_data *)client_data;

    auto log = toast::Logger::get();
    std::ostringstream o;
    o << "Stream decode error (" << status << ") at input byte range ";
    o << data->in_offset << " - " << data->in_end << ", output size = ";
    o << data->output->size();
    log.error(o.str().c_str());
    throw std::runtime_error(o.str().c_str());
    return;
}

void decode_flac(
    uint8_t * const bytes,
    size_t n_bytes,
    int64_t * const offsets,
    size_t n_offset,
    toast::AlignedI32 & data
) {
    dec_callback_data callback_data;
    callback_data.input = bytes;
    callback_data.in_nbytes = n_bytes;
    callback_data.output = &data;

    FLAC__StreamDecoder * decoder;
    bool success;
    FLAC__StreamDecoderInitStatus status;

    size_t n_sub = n_offset;

    for (size_t sub = 0; sub < n_sub; ++sub) {
        callback_data.in_offset = offsets[sub];
        if (sub == n_sub - 1) {
            callback_data.in_end = n_bytes;
        } else {
            callback_data.in_end = offsets[sub + 1];
        }

        // std::cerr << "Decoding chunk " << sub << " at byte offset " <<
        // callback_data.in_offset << " with " << (callback_data.in_end -
        // callback_data.in_offset) << " bytes" << std::endl;

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
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Failed to initialize decoder, status = " << status;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }

        success = FLAC__stream_decoder_process_until_end_of_stream(decoder);
        if (!success) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Failed on decoder_process_until_end_of_stream for chunk " << sub;
            o << ", byte range " << callback_data.in_offset << " - ";
            o << callback_data.in_end;
            o << ", output size = " << callback_data.output->size();
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }

        success = FLAC__stream_decoder_finish(decoder);
        if (!success) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Failed on decoder_finish";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }

        FLAC__stream_decoder_delete(decoder);
    }

    return;
}

#endif // ifdef HAVE_FLAC


void init_io_compression_flac(py::module & m) {
    // FLAC compression

    m.def(
        "have_flac_support", []() {
            #ifdef HAVE_FLAC
            return true;

            #else // ifdef HAVE_FLAC
            return false;

            #endif // ifdef HAVE_FLAC
        }, R"(
        Return True if TOAST is compiled with FLAC support.
    )");

    m.def(
        "compress_flac_2D", [](
            py::buffer data,
            uint32_t level
        ) {
            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            int32_t * raw_data = extract_buffer <int32_t> (
                data, "int32 data", 2, temp_shape, {-1, -1}
            );
            int64_t n_chunk = temp_shape[0];
            int64_t n_chunk_elem = temp_shape[1];

            toast::AlignedU8 bytes;
            toast::AlignedI64 offsets;

            #ifdef HAVE_FLAC
            encode_flac(
                raw_data,
                n_chunk * n_chunk_elem,
                bytes,
                offsets,
                level,
                n_chunk_elem
            );
            #else // ifdef HAVE_FLAC
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "TOAST was not built with libFLAC support";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
            #endif // ifdef HAVE_FLAC

            // std::cout << "compress_flac_2D returning buffer @ " <<
            // (int64_t)bytes.data() << std::endl;

            return py::make_tuple(py::cast(bytes), py::cast(offsets));
        }, py::arg("data"), py::arg(
            "level"), R"(
        Compress 2D 32bit integer data with FLAC.

        Each row of the input is compressed separately, and the byte offset
        into the output stream is returned.

        Args:
            data (array, int32):  The 2D array of integer data.
            level (uint32):  The compression level (0-8).

        Returns:
            (tuple):  The (byte array, offsets).

    )");

    m.def(
        "decompress_flac_2D", [](
            py::buffer data,
            py::buffer offsets
        ) {
            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            uint8_t * raw_data = extract_buffer <uint8_t> (
                data, "FLAC bytes", 1, temp_shape, {-1}
            );
            int64_t n_bytes = temp_shape[0];

            int64_t * raw_offsets = extract_buffer <int64_t> (
                offsets, "FLAC offsets", 1, temp_shape, {-1}
            );
            int64_t n_offset = temp_shape[0];

            toast::AlignedI32 output;

            #ifdef HAVE_FLAC
            decode_flac(
                raw_data,
                n_bytes,
                raw_offsets,
                n_offset,
                output
            );
            #else // ifdef HAVE_FLAC
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "TOAST was not built with libFLAC support";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
            #endif // ifdef HAVE_FLAC

            // std::cout << "decompress_flac_2D returning buffer @ " <<
            // (int64_t)output.data() << std::endl;

            return py::cast(output);
        }, py::arg("data"), py::arg(
            "offsets"), R"(
        Decompress FLAC bytes into 2D 32bit integer data.

        The array of bytes is decompressed and returned.

        Args:
            data (array, uint8):  The 1D array of bytes.
            offsets (array, int64):  The array of offsets into the byte array.

        Returns:
            (array):  The array of 32bit integers.

    )");

    m.def(
        "compress_flac", [](
            py::buffer data,
            uint32_t level
        ) {
            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            int32_t * raw_data = extract_buffer <int32_t> (
                data, "int32 data", 1, temp_shape, {-1}
            );
            int64_t n = temp_shape[0];

            toast::AlignedU8 bytes;
            toast::AlignedI64 offsets;

            #ifdef HAVE_FLAC
            encode_flac(
                raw_data,
                n,
                bytes,
                offsets,
                level,
                0
            );
            #else // ifdef HAVE_FLAC
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "TOAST was not built with libFLAC support";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
            #endif // ifdef HAVE_FLAC

            // std::cout << "compress_flac returning buffer @ " << (int64_t)bytes.data()
            // << std::endl;

            return py::cast(bytes);
        }, py::arg("data"), py::arg(
            "level"), R"(
        Compress 1D 32bit integer data with FLAC.

        The 1D array is compressed and the byte array is returned.

        Args:
            data (array, int32):  The 1D array of integer data.
            level (uint32):  The compression level (0-8).

        Returns:
            (array):  The byte array.

    )");

    m.def(
        "decompress_flac", [](
            py::buffer data
        ) {
            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            uint8_t * raw_data = extract_buffer <uint8_t> (
                data, "FLAC bytes", 1, temp_shape, {-1}
            );
            int64_t n_bytes = temp_shape[0];

            toast::AlignedI32 output;

            int64_t offset = 0;

            #ifdef HAVE_FLAC
            decode_flac(
                raw_data,
                n_bytes,
                &offset,
                1,
                output
            );
            #else // ifdef HAVE_FLAC
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "TOAST was not built with libFLAC support";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
            #endif // ifdef HAVE_FLAC

            // std::cout << "decompress_flac returning buffer @ " <<
            // (int64_t)output.data() << std::endl;

            return py::cast(output);
        }, py::arg(
            "data"), R"(
        Decompress FLAC bytes into 1D 32bit integer data.

        The array of bytes is decompressed and returned.

        Args:
            data (array, uint8):  The 1D array of bytes.

        Returns:
            (array):  The array of 32bit integers.

    )");

    return;
}
