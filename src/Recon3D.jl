using CUDA

include("LineOfResponse.jl")
include("Util.jl")

struct Slice
    is_valid::Bool
    i1::Int32
    i2::Int32
    i3::Int32
    i4::Int32
    l_min_min::Float32
    l_min_plus::Float32
    l_plus_min::Float32
    l_plus_plus::Float32
end

# Calculate index of block-major order array
@inline function calculate_index(i, j, k, DIMX, DIMY, DIMZ, bs)
	DIMX = fld1(DIMX, bs)*bs
	DIMY = fld1(DIMY, bs)*bs
	DIMZ = fld1(DIMZ, bs)*bs

    block_i, intra_block_i = fldmod(i-Int32(1), bs)
    block_j, intra_block_j = fldmod(j-Int32(1), bs)
    block_k, intra_block_k = fldmod(k-Int32(1), bs)

    x = bs*bs*bs*block_i + intra_block_i
    y = DIMX*bs*bs*block_j + bs*intra_block_j
    z = DIMX*DIMY*bs*block_k + bs*bs*intra_block_k

    return unsafe_trunc(Int32, x + y + z + 1)
end

@inline function calculate_length(DIMX, DIMY, DIMZ, bs)
    X = fld1(DIMX, bs) * bs
    Y = fld1(DIMY, bs) * bs
    Z = fld1(DIMZ, bs) * bs

    return X*Y*Z
end

@inline function transform_dimensions(plane::Plane, DIMX, DIMY, DIMZ)
    # Transform event so we can use the same code for different main planes.
    if plane == x_plane
        return DIMX, DIMY, DIMZ
    elseif plane == y_plane
        return DIMY, DIMX, DIMZ
    else
        return DIMZ, DIMY, DIMX
    end
end

# Reduce a value across a warp
@inline function reduce_warp(op, val)
    # offset = warpsize() ÷ 2
    # while offset > 0
    #     val = op(val, shfl_down_sync(0xffffffff, val, offset))
    #     offset ÷= 2
    # end

    # Loop unrolling for warpsize = 32
    val = op(val, shfl_down_sync(0xffffffff, val, 16, 32))
    val = op(val, shfl_down_sync(0xffffffff, val, 8, 32))
    val = op(val, shfl_down_sync(0xffffffff, val, 4, 32))
    val = op(val, shfl_down_sync(0xffffffff, val, 2, 32))
    val = op(val, shfl_down_sync(0xffffffff, val, 1, 32))

    return val
end

# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(op, val::T, neutral) where T
    # shared mem for 32 partial sums
    shared = @cuStaticSharedMem(T, 32)  # NOTE: this is an upper bound; better detect it

    wid, lane = fldmod1(threadIdx().x, warpsize())

    # each warp performs partial reduction
    val = reduce_warp(op, val)

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end

    # wait for all partial reductions
    sync_threads()

    # read from shared memory only if that warp existed
    val = if threadIdx().x <= fld1(blockDim().x, warpsize())
         @inbounds shared[lane]
    else
        neutral
    end

    # final reduce within first warp
    if wid == 1
        val = reduce_warp(op, val)
    end

    return val
end

@inline function ray_tracing(event::Event, t, DIMX::Int32, DIMY::Int32, DIMZ::Int32)
    x1 = event.lor.P1.x / 3.125f0
    x2 = event.lor.P2.x / 3.125f0
    y1 = event.lor.P1.y / 3.125f0
    y2 = event.lor.P2.y / 3.125f0
    z1 = event.lor.P1.z / 3.125f0
    z2 = event.lor.P2.z / 3.125f0

    k_y = (y2 - y1) / (x2 - x1)
    k_z = (z2 - z1) / (x2 - x1)

    i = t - DIMX / 2.0f0 - 1.0f0

    if k_y >= 0
        y_min = y1 + k_y * (i - x1) + DIMY / 2.0f0
        y_plus = y1 + k_y * (i + 1.0f0 - x1) + DIMY / 2.0f0
    else
        y_min = y1 + k_y * (i + 1.0f0 - x1) + DIMY / 2.0f0
        y_plus = y1 + k_y * (i - x1) + DIMY / 2.0f0
    end

    Y_min = floor(Int32, y_min) # Zero based indices
    Y_plus = floor(Int32, y_plus) # Zero based indices

    if k_z >= 0
        z_min = z1 + k_z * (i - x1) + DIMZ / 2.0f0
        z_plus = z1 + k_z * (i + 1.0f0 - x1) + DIMZ / 2.0f0
    else
        z_min = z1 + k_z * (i + 1.0f0 - x1) + DIMZ / 2.0f0
        z_plus = z1 + k_z * (i - x1) + DIMZ / 2.0f0
    end

    Z_min = floor(Int32, z_min) # Zero based indices
    Z_plus = floor(Int32, z_plus) # Zero based indices

    l = sqrt(1.0f0 + (y_plus - y_min)^2 + (z_plus - z_min)^2)

    l_min_min = 0.0f0
    l_min_plus = 0.0f0
    l_plus_min = 0.0f0
    l_plus_plus = 0.0f0

    if Y_min == Y_plus
        if Y_min >= 0 && Y_min < DIMY
            if Z_min == Z_plus
                # 1-voxel case
                if Z_min >= 0 && Z_min < DIMZ
                    l_min_min = l
                end
            else
                # 2-voxel case in z-direction
                if Z_min >= -1 && Z_min < DIMZ
                    r_z = (Z_plus - z_min) / (z_plus - z_min)

                    if Z_min > -1
                        l_min_min = l * r_z
                    end
                    if Z_plus < DIMZ
                        l_min_plus = l * (1.0f0 - r_z)
                    end
                end
            end
        end
    else
        if Y_min >= -1 && Y_min < DIMY
            if Z_min == Z_plus
                # 2-voxel case in y-direction
                if Z_min >= 0 && Z_min < DIMZ
                    r_y = (Y_plus - y_min) / (y_plus - y_min)

                    if Y_min > -1
                        l_min_min = l * r_y
                    end
                    if Y_plus < DIMY
                        l_plus_min = l * (1.0f0 - r_y)
                    end
                end
            else
                # 3-voxel cases
                if Z_min >= -1 && Z_min < DIMZ
                    r_y = (Y_plus - y_min) / (y_plus - y_min)
                    r_z = (Z_plus - z_min) / (z_plus - z_min)

                    if r_y > r_z
                        if Y_min > -1 && Z_min > -1
                            l_min_min = l * r_z
                        end
                        if Y_min > -1 && Z_plus < DIMZ
                            l_min_plus = l * (r_y - r_z)
                        end
                        if Y_plus < DIMY && Z_plus < DIMZ
                            l_plus_plus = l * (1.0f0 - r_y)
                        end
                    else
                        if Y_min > -1 && Z_min > -1
                            l_min_min = l * r_y
                        end
                        if Y_plus < DIMY && Z_min > -1
                            l_plus_min = l * (r_z - r_y)
                        end
                        if Y_plus < DIMY && Z_plus < DIMZ
                            l_plus_plus = l * (1.0f0 - r_z)
                        end
                    end
                end
            end
        end
    end

    return Y_min, Y_plus, Z_min, Z_plus, l_min_min, l_min_plus, l_plus_min, l_plus_plus
end

@inline function get_element(thread_i, index, array::CuDeviceArray{T}) where {T}
    shared = @cuStaticSharedMem(T, 1)
    if thread_i == 1
        shared[1] = array[index]
    end
    CUDA.sync_threads()
    element = shared[1]

    return element
end

@inline function perm(main_plane::Plane, x, y, z)
	x = unsafe_trunc(Int32, x)
	y = unsafe_trunc(Int32, y)
	z = unsafe_trunc(Int32, z)
    if main_plane == x_plane
        return x, y, z
    elseif main_plane == y_plane
        return y, x, z
    else
        return z, y, x
    end
end

@inline @inbounds function forward_project(thread_i, event::Event, image, DIMX, DIMY, DIMZ)
    block_size = Int32(4)
    one = Int32(1)

    DIMX_event, DIMY_event, DIMZ_event = transform_dimensions(event.mainPlane, DIMX, DIMY, DIMZ)

    nr_of_iterations = fld1(DIMX_event - (thread_i - one), blockDim().x)
    slices = @cuDynamicSharedMem(Slice, DIMX_event)
    value = 0.0f0

    for i = one:nr_of_iterations
        t = Int32(thread_i + (i - one) * blockDim().x)

        # Ray tracing
        Y_min, Y_plus, Z_min, Z_plus, l_min_min, l_min_plus, l_plus_min, l_plus_plus = ray_tracing(event, t, DIMX_event, DIMY_event, DIMZ_event)

        i1 = calculate_index(perm(event.mainPlane, t, Y_min + 1, Z_min + 1)..., DIMX, DIMY, DIMZ, 4)
        i2 = calculate_index(perm(event.mainPlane, t, Y_min + 1, Z_plus + 1)..., DIMX, DIMY, DIMZ, 4)
        i3 = calculate_index(perm(event.mainPlane, t, Y_plus + 1, Z_min + 1)..., DIMX, DIMY, DIMZ, 4)
        i4 = calculate_index(perm(event.mainPlane, t, Y_plus + 1, Z_plus + 1)..., DIMX, DIMY, DIMZ, 4)

        if Y_min >= 0 && Z_min >= 0 && Y_plus < DIMY_event && Z_plus < DIMZ_event
        # if Y_plus >= 0 && Z_plus >=0 && Y_min < DIMY_event && Z_min < DIMZ_event
            value += image[i1] * l_min_min
            value += image[i2] * l_min_plus
            value += image[i3] * l_plus_min
            value += image[i4] * l_plus_plus

            slices[t] = Slice(true, i1, i2, i3, i4, l_min_min, l_min_plus, l_plus_min, l_plus_plus)
        else
            slices[t] = Slice(false, i1, i2, i3, i4, l_min_min, l_min_plus, l_plus_min, l_plus_plus)
        end
    end

    # Forward project

    # Calculate total ray length
    total_value = reduce_block(+, value, 0.0f0)

    # Share total length across all threads in the block
    shared_total_value = @cuStaticSharedMem(Float32, 1)
    if thread_i == 1
        shared_total_value[1] = total_value
    end
    CUDA.sync_threads()
    total_value = shared_total_value[1]

    return total_value, nr_of_iterations, slices
end

@inline @inbounds function back_project(thread_i, nr_of_iterations, slices, total_value, corr)
    for i = Int32(1):nr_of_iterations
        t = Int32(thread_i + (i - Int32(1)) * blockDim().x)
        slice = slices[t]

        if slice.is_valid && total_value > 0
            corr[slice.i1] += slice.l_min_min / total_value
            corr[slice.i2] += slice.l_min_plus / total_value
            corr[slice.i3] += slice.l_plus_min / total_value
            corr[slice.i4] += slice.l_plus_plus / total_value
        end
    end
end

function gpu_kernel(events::CuDeviceArray{Event}, image::CuDeviceArray{T,1}, corr::CuDeviceArray{T,1}, DIMX::Int32, DIMY::Int32, DIMZ::Int32) where {T}
    thread_i = threadIdx().x
    index = blockIdx().x

    event = get_element(thread_i, index, events)
    total_value, nr_of_iterations, slices = forward_project(thread_i, event, image, DIMX, DIMY, DIMZ)

    back_project(thread_i, nr_of_iterations, slices, total_value, corr)

    return nothing
end

# No extensions
function reconstruct3D(events, DIMX, DIMY, DIMZ, recon_iters)
    # Transfer the events from CPU memory to GPU memory
    c_events = CUDA.CuArray(events);

    # Initialise the image array on GPU
    c_image = CUDA.ones(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));
  
    n_threads = min(max(DIMX, DIMY, DIMZ), 128)
    n_blocks = length(c_events)
    n_shmem = max(DIMX, DIMY, DIMZ) * sizeof(Slice)

    for k = 1:recon_iters
        # Initialise correction matrix memory on GPU
        c_corr = CUDA.zeros(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));

        # Schedule a kernel that performs:
        #   - Forward projection
        #   - Compare
        #   - Back projection
        @cuda blocks=n_blocks threads=n_threads shmem=n_shmem gpu_kernel(c_events, c_image, c_corr, Int32(DIMX), Int32(DIMY), Int32(DIMZ))

        # Perform the update step
        c_image = c_image .* c_corr
    end

    # Transfer the image estimate from GPU to CPU memory
    image = Array(c_image)

    return image
end

# Sensitivity correction
function reconstruct3D(events, sensmap, DIMX, DIMY, DIMZ, recon_iters)
    # Transfer the events from CPU memory to GPU memory
    c_events = CUDA.CuArray(events);

    # Initialise the image array on GPU
    c_image = CUDA.ones(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));

    # Initialise the sensmap array on GPU
    c_sensmap = CUDA.CuArray(sensmap);

    n_threads = min(max(DIMX, DIMY, DIMZ), 128)
    n_blocks = length(c_events)
    n_shmem = max(DIMX, DIMY, DIMZ) * sizeof(Slice)

    for k = 1:recon_iters
        # Initialise correction matrix memory on GPU
        c_corr = CUDA.zeros(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));

        # Schedule a kernel that performs:
        #   - Forward projection
        #   - Compare
        #   - Back projection
        @cuda blocks=n_blocks threads=n_threads shmem=n_shmem gpu_kernel(c_events, c_image, c_corr, Int32(DIMX), Int32(DIMY), Int32(DIMZ))

        # Perform the update step
        c_image = c_image .* c_corr ./ c_sensmap
    end

    # Transfer the image estimate from GPU to CPU memory
    image = Array(c_image)

    return image
end

# OSEM with sensitivity correction
function OS_reconstruct3D(events, sensmap, DIMX, DIMY, DIMZ, recon_iters, number_of_subsets)
	# Transfer the events per subset from CPU memory to GPU memory
	c_events = [CUDA.CuArray(events[i:number_of_subsets:length(events)]) for i in 1:number_of_subsets]

	# Initialise the sensmap array on GPU
	c_sensmap = CUDA.CuArray(sensmap);

	# Initialise the image array on GPU
	c_image = CUDA.ones(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));

	n_threads = min(max(DIMX, DIMY, DIMZ), 224)
	n_shmem = max(DIMX, DIMY, DIMZ)*sizeof(Slice)

	for k = 1:recon_iters
		for subset = 1:number_of_subsets
			# Initialise correction matrix memory on GPU
			c_corr = CUDA.zeros(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));

			n_blocks = length(c_events[subset])

			# Schedule a kernel that performs:
			#   - Forward projection
			#   - Compare
			#   - Back projection
			@cuda blocks=n_blocks threads=n_threads shmem=n_shmem gpu_kernel(c_events[subset], c_image, c_corr, Int32(DIMX), Int32(DIMY), Int32(DIMZ))

			# Perform the update step
			c_image = c_image .* (c_corr * number_of_subsets) ./ c_sensmap
		end
	end

	# Transfer the image estimate from GPU to CPU memory
	image = Array(c_image)

	return image
end