struct Point
	x::Float32
	y::Float32
	z::Float32
end

@enum Plane begin
	undef_plane
    x_plane
    y_plane
	z_plane
end

struct LOR
	P1::Point
	P2::Point
	timeDiff::Float32
end

struct Event
	lor::LOR
	mainPlane::Plane
end
