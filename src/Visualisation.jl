using CUDA

# Detect whether there is a display available
display_mode = true
try
   run(pipeline(`xset -q`, `tail -n1`, `grep "Monitor is On"`));
catch e
    global display_mode = false
end
if display_mode
    using ImageView, Images
end


function visualise(c_image::CuArray{Float32,2})
    image = Array(c_image)

    display_mode ? imshow(image) : println("No display available")
end

function visualise(c_image::CuArray{Float32,3})
    image = Array(c_image)

    display_mode ? imshow(image) : println("No display available")
end


function visualise(c_image::CuArray{Float32,1}, DIMX::Integer, DIMY::Integer, DIMZ::Integer)
    image = Array(c_image);

    visualise(image, DIMX, DIMY, DIMZ)
end

function visualise(image::Array{Float32,1}, DIMX::Integer, DIMY::Integer, DIMZ::Integer)
    output = zeros(Float32, DIMX, DIMY, DIMZ);
    for i = 1:DIMX
        for j = 1:DIMY
            for k = 1:DIMZ
                index = calculate_index(i, j, k, DIMX, DIMY, DIMZ, 4)
                output[i, j, k] = image[index]
            end
        end
    end

    display_mode ? imshow(output) : println("No display available")
end


function visualise(cpp_image::Ptr{Float32}, DIMX::Integer, DIMY::Integer, DIMZ::Integer)
    image = unsafe_wrap(Array, cpp_image, (DIMX, DIMY, DIMZ))

    display_mode ? imshow(image) : println("No display available")
end

function visualise(image::Array{Float32,3})
    display_mode ? imshow(image) : println("No display available")
end

function visualise(image::Array{Float64,3})
    display_mode ? imshow(image) : println("No display available")
end

function visualise(image::Array{Float32,2})
    display_mode ? imshow(image) : println("No display available")
end

function save_image(c_image::CuArray{Float32,1}, DIMX::Integer, DIMY::Integer, DIMZ::Integer)
    image = Array(c_image);

    save_image(image, DIMX, DIMY, DIMZ)
end

function save_image(image::Array{Float32,1}, DIMX::Integer, DIMY::Integer, DIMZ::Integer)
    output = zeros(Float32, DIMX, DIMY, DIMZ);
    for i = 1:DIMX
        for j = 1:DIMY
            for k = 1:DIMZ
                index = calculate_index(i, j, k, DIMX, DIMY, DIMZ, 4)
                output[i, j, k] = image[index]
            end
        end
    end

    save_image(output)
end

function save_image(image::Array{Float32,1})
    io = open("recon.raw","w")

    for i in eachindex(image)
        write(io, image[i])
    end

    close(io)
end

function save_image(image::Array{Float32,3})
    save_image(image[:])
end
