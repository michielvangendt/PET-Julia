using CUDA

include("LineOfResponse.jl")
include("Util.jl")

function ray_tracing(event::Event, t, DIMX, DIMY, DIMZ)
    x1 = event.lor.P1.x
    x2 = event.lor.P2.x
    y1 = event.lor.P1.y
    y2 = event.lor.P2.y
    z1 = event.lor.P1.z
    z2 = event.lor.P2.z

    k_y = (y2 - y1) / (x2 - x1)
    k_z = (z2 - z1) / (x2 - x1)

    i = t - DIMX/2 - 1

    if k_y >= 0
        y_min = y1 + k_y*(i-x1) + DIMY/2
        y_plus = y1 + k_y*(i+1-x1) + DIMY/2
    else
        y_min = y1 + k_y*(i+1-x1) + DIMY/2
        y_plus = y1 + k_y*(i-x1) + DIMY/2
    end

    Y_min = floor(Int32, y_min) # Zero based indices
    Y_plus = floor(Int32, y_plus) # Zero based indices

    if k_z >= 0
        z_min = z1 + k_z*(i-x1) + DIMZ/2
        z_plus = z1 + k_z*(i+1-x1) + DIMZ/2
    else
        z_min = z1 + k_z*(i+1-x1) + DIMZ/2
        z_plus = z1 + k_z*(i-x1) + DIMZ/2
    end

    Z_min = floor(Int32, z_min) # Zero based indices
    Z_plus = floor(Int32, z_plus) # Zero based indices

    l = sqrt(1 + (y_plus - y_min)^2 + (z_plus - z_min)^2)

    l_min_min = 0.0
    l_min_plus = 0.0
    l_plus_min = 0.0
    l_plus_plus = 0.0

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
                    r_z = (Z_plus - z_min)/(z_plus - z_min)

                    if Z_min > -1
                        l_min_min = l * r_z
                    end
                    if Z_plus < DIMZ
                        l_min_plus = l * (1 - r_z)
                    end
                end
            end
        end
    else
        if Y_min >= -1 && Y_min < DIMY
            if Z_min == Z_plus
                # 2-voxel case in y-direction
                if Z_min >= 0 && Z_min < DIMZ
                    r_y = (Y_plus - y_min)/(y_plus - y_min)

                    if Y_min > -1
                        l_min_min = l * r_y
                    end
                    if Y_plus < DIMY
                        l_plus_min = l * (1 - r_y)
                    end
                end
            else
                # 3-voxel cases
                if Z_min >= -1 && Z_min < DIMZ
                    r_y = (Y_plus - y_min)/(y_plus - y_min)
                    r_z = (Z_plus - z_min)/(z_plus - z_min)

                    if r_y > r_z
                        if Y_min > -1 && Z_min > -1
                            l_min_min = l * r_z
                        end
                        if Y_min > -1 && Z_plus < DIMZ
                            l_min_plus = l * (r_y - r_z)
                        end
                        if Y_plus < DIMY && Z_plus < DIMZ
                            l_plus_plus = l * (1 - r_y)
                        end
                    else
                        if Y_min > -1 && Z_min > -1
                            l_min_min = l * r_y
                        end
                        if Y_plus < DIMY && Z_min > -1
                            l_plus_min = l * (r_z - r_y)
                        end
                        if Y_plus < DIMY && Z_plus < DIMZ
                            l_plus_plus = l * (1 - r_z)
                        end
                    end
                end
            end
        end
    end

    return Y_min, Y_plus, Z_min, Z_plus, l_min_min, l_min_plus, l_plus_min, l_plus_plus
end

function perm(main_plane::Plane, x, y, z)
	if main_plane == x_plane
		return x, y, z
	elseif main_plane == y_plane
		return y, x, z
	else
		return z, y, x
	end
end

function gpu_kernel(events::CuDeviceArray{Event}, image::CuDeviceArray{T,3}, corr::CuDeviceArray{T,3}, tmp_total_values::CuDeviceArray{T,1}, DIMX, DIMY, DIMZ) where {T}
    t = threadIdx().x
    index = blockIdx().x

    event = events[index]

    # Ray tracing
    Y_min, Y_plus, Z_min, Z_plus, l_min_min, l_min_plus, l_plus_min, l_plus_plus = ray_tracing(event, t, DIMX, DIMY, DIMZ)

    if Y_min >= 0 && Z_min >=0 && Y_plus < DIMY && Z_plus < DIMZ
        value = 0.0
		value += image[perm(event.mainPlane, t, Y_min+1, Z_min+1)...] * l_min_min
		value += image[perm(event.mainPlane, t, Y_min+1, Z_plus+1)...] * l_min_plus
		value += image[perm(event.mainPlane, t, Y_plus+1, Z_min+1)...] * l_plus_min
		value += image[perm(event.mainPlane, t, Y_plus+1, Z_plus+1)...] * l_plus_plus

        # Forward project
        CUDA.atomic_add!(pointer(tmp_total_values, index), Float32(value))
        CUDA.sync_threads()
        total_value = tmp_total_values[index]

        if total_value > 0
            # Compare and back project
			corr[perm(event.mainPlane, t, Y_min+1, Z_min+1)...] += l_min_min / total_value
            corr[perm(event.mainPlane, t, Y_min+1, Z_plus+1)...] += l_min_plus / total_value
            corr[perm(event.mainPlane, t, Y_plus+1, Z_min+1)...] += l_plus_min / total_value
            corr[perm(event.mainPlane, t, Y_plus+1, Z_plus+1)...] += l_plus_plus / total_value
        end
    end

    return nothing
end

function reconstruct3D(events, DIMX, DIMY, DIMZ, recon_iters)
  # Transfer the events from CPU memory to GPU memory
  c_events = CUDA.CuArray(events);

  # Initialise the image array on GPU
  c_image = CUDA.ones(Float32, DIMX, DIMY, DIMZ);

  for k = 1:recon_iters
    # Initialise correction matrix memory on GPU
    c_corr = CUDA.zeros(Float32, DIMX, DIMY, DIMZ);
    # Initialise array to store forward projection values on GPU
    c_tmp_total_values = CUDA.zeros(Float32, length(c_events))

    # Schedule a kernel that performs:
    #   - Forward projection
    #   - Compare
    #   - Back projection
    @cuda blocks=length(c_events) threads=DIMX gpu_kernel(c_events, c_image, c_corr, c_tmp_total_values, DIMX, DIMY, DIMZ)

    # Perform the update step
    c_image = c_image .* c_corr
  end

  # Transfer the image estimate from GPU to CPU memory
  image = Array(c_image)

  return image
end
