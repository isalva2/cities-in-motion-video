import geopandas as gpd
import folium
import webbrowser

shp_path = r"C:\Users\joaob\Dropbox\Documents\hackaton_UIC\hackaton_project\data\SAGE_node_shapefiles\node_W06E.shp"

# Load in original CRS (EPSG:26916) to extract UTM coords too
gdf_utm = gpd.read_file(shp_path)  # native EPSG:26916

# Reproject to WGS84 for folium display
gdf = gdf_utm.to_crs(epsg=4326)
center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]

m = folium.Map(
    location=center,
    zoom_start=20,
    max_zoom=22,
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google"
)

folium.GeoJson(
    gdf,
    name="node_W042",
    style_function=lambda x: {
        "color": "#ffff00",
        "weight": 3,
        "fillOpacity": 0.2,
        "fillColor": "#ffff00"
    },
    tooltip=folium.GeoJsonTooltip(fields=list(gdf.columns[:-1]))
).add_to(m)

# ── Click-to-coordinate panel ─────────────────────────────────────────────────
click_js = """
<style>
    #coord-panel {
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background: rgba(0,0,0,0.85);
        color: #00ff88;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        padding: 12px 20px;
        border-radius: 8px;
        border: 1px solid #00ff88;
        min-width: 520px;
        text-align: center;
        pointer-events: none;
    }
    #coord-panel b { color: #ffff00; }
    #gcps-log {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 9999;
        background: rgba(0,0,0,0.90);
        color: #ffffff;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #3498db;
        max-width: 340px;
        max-height: 400px;
        overflow-y: auto;
    }
    #gcps-log h4 { color: #3498db; margin: 0 0 6px 0; }
    #gcps-log .gcp-entry { color: #aaffaa; margin: 2px 0; }
    #copy-btn {
        margin-top: 8px;
        background: #3498db;
        color: white;
        border: none;
        padding: 4px 10px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 11px;
        pointer-events: all;
    }
</style>

<div id="coord-panel">🖱️ Click anywhere on the map to get coordinates</div>

<div id="gcps-log">
    <h4>📍 GCP Log (EPSG:26916)</h4>
    <div id="gcp-list">No clicks yet...</div>
    <button id="copy-btn" onclick="copyGCPs()">📋 Copy GCPs</button>
</div>

<script>
var gcpList = [];
var clickCount = 0;

// Proj4 for converting WGS84 → UTM Zone 16N (EPSG:26916)
var script = document.createElement('script');
script.src = 'https://cdnjs.cloudflare.com/ajax/libs/proj4js/2.9.0/proj4.js';
script.onload = function() {
    proj4.defs("EPSG:26916", "+proj=utm +zone=16 +datum=NAD83 +units=m +no_defs");

    // Attach click listener AFTER proj4 loads
    var map = Object.values(window).find(v => v && v._container && v.on);
    if (!map) {
        // fallback: find leaflet map
        for (var key in window) {
            try {
                if (window[key] && window[key]._leaflet_id) { map = window[key]; break; }
            } catch(e) {}
        }
    }

    document.querySelector('.leaflet-container').addEventListener('click', function(e) {
        // Get lat/lon from leaflet map instance
        var allMaps = [];
        for (var key in window) {
            try { if (window[key] && window[key].getCenter) allMaps.push(window[key]); }
            catch(e) {}
        }
        var leafMap = allMaps[0];
        if (!leafMap) return;

        var latlng = leafMap.mouseEventToLatLng(e);
        var lat = latlng.lat;
        var lon = latlng.lng;

        // Convert to EPSG:26916 (UTM Zone 16N)
        var utm = proj4("EPSG:4326", "EPSG:26916", [lon, lat]);
        var utmX = utm[0].toFixed(3);
        var utmY = utm[1].toFixed(3);

        clickCount++;

        // Update bottom panel
        document.getElementById('coord-panel').innerHTML =
            '📍 Click #' + clickCount +
            ' &nbsp;|&nbsp; <b>Lat:</b> ' + lat.toFixed(7) +
            ' &nbsp;<b>Lon:</b> ' + lon.toFixed(7) +
            ' &nbsp;|&nbsp; <b>UTM X:</b> ' + utmX +
            ' &nbsp;<b>UTM Y:</b> ' + utmY;

        // Log to GCP panel
        gcpList.push([parseFloat(utmX), parseFloat(utmY)]);
        var listDiv = document.getElementById('gcp-list');
        listDiv.innerHTML = gcpList.map(function(p, i) {
            return '<div class="gcp-entry">GCP ' + (i+1) +
                   ': [' + p[0].toFixed(2) + ', ' + p[1].toFixed(2) + ']</div>';
        }).join('');

        // Drop a marker
        L.circleMarker([lat, lon], {
            radius: 7,
            color: '#ff4444',
            fillColor: '#ffff00',
            fillOpacity: 1,
            weight: 2
        }).addTo(leafMap).bindPopup(
            '<b>GCP ' + clickCount + '</b><br>' +
            'UTM X: ' + utmX + '<br>' +
            'UTM Y: ' + utmY
        ).openPopup();
    });
};
document.head.appendChild(script);

function copyGCPs() {
    if (gcpList.length === 0) { alert('No GCPs yet!'); return; }
    var text = 'WORLD_POINTS = np.float32([\\n';
    gcpList.forEach(function(p, i) {
        text += '    [' + p[0].toFixed(3) + ', ' + p[1].toFixed(3) + '],   # GCP ' + (i+1) + '\\n';
    });
    text += '])';
    navigator.clipboard.writeText(text).then(function() {
        alert('✅ Copied! Paste directly into your Python script.');
    });
}
</script>
"""

m.get_root().html.add_child(folium.Element(click_js))
folium.LayerControl().add_to(m)

out_path = r"C:\Users\joaob\Dropbox\Documents\hackaton_UIC\hackaton_project\data\SAGE_node_shapefiles\node_W042_map.html"
m.save(out_path)
webbrowser.open(out_path)
print("Map saved! Click on intersection features to collect GCP coordinates.")