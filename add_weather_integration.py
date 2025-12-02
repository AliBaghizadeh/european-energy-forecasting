"""
Script to add weather API integration to energy_app.py
"""

# Read the original file
with open("App/energy_app_backup.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find and modify specific sections
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # 1. Add weather_api import after csv import
    if line.strip() == "import csv":
        new_lines.append(line)
        new_lines.append("from weather_api import fetch_current_temperature\n")
        i += 1
        continue
    
    # 2. Add weather fetch function before "# Gradio Interface"
    if line.strip() == "# Gradio Interface":
        # Add weather function
        new_lines.append("\n")
        new_lines.append("# Weather API Integration\n")
        new_lines.append("def fetch_weather_for_country(country_id: str):\n")
        new_lines.append("    \"\"\"\n")
        new_lines.append("    Fetch current weather for a country and return temperature + status message.\n")
        new_lines.append("    \n")
        new_lines.append("    Returns:\n")
        new_lines.append("        tuple: (temperature: float, status_message: str)\n")
        new_lines.append("    \"\"\"\n")
        new_lines.append("    weather_data = fetch_current_temperature(country_id)\n")
        new_lines.append("    \n")
        new_lines.append("    if weather_data is None:\n")
        new_lines.append("        return (\n")
        new_lines.append("            15.0,  # Default fallback temperature\n")
        new_lines.append("            \"⚠️ Could not fetch weather data. Using default temperature.\",\n")
        new_lines.append("        )\n")
        new_lines.append("    \n")
        new_lines.append("    temp = weather_data[\"temperature\"]\n")
        new_lines.append("    city = weather_data[\"city\"]\n")
        new_lines.append("    description = weather_data[\"description\"]\n")
        new_lines.append("    \n")
        new_lines.append("    status_msg = f\"✅ Fetched from {city}: {temp}°C ({description})\"\n")
        new_lines.append("    \n")
        new_lines.append("    return temp, status_msg\n")
        new_lines.append("\n\n")
        new_lines.append(line)  # Add the "# Gradio Interface" line
        i += 1
        continue
    
    # 3. Modify UI section - reorder inputs and add weather button
    if "last_load = gr.Slider(" in line:
        # Add last_load slider
        new_lines.append(line)
        i += 1
        # Skip to end of last_load definition
        while i < len(lines) and ")" not in lines[i]:
            new_lines.append(lines[i])
            i += 1
        new_lines.append(lines[i])  # Add closing paren
        i += 1
        new_lines.append("                    \n")
        
        # Skip current_temp slider (we'll add it later)
        while i < len(lines) and "current_temp = gr.Slider(" not in lines[i]:
            i += 1
        while i < len(lines) and ")" not in lines[i]:
            i += 1
        i += 1  # Skip the closing paren
        
        # Add country_id radio
        while i < len(lines) and "country_id = gr.Radio(" not in lines[i]:
            i += 1
        while i < len(lines) and "predict_btn = gr.Button(" not in lines[i]:
            new_lines.append(lines[i])
            i += 1
        
        # Add weather button and status
        new_lines.append("                    \n")
        new_lines.append("                    with gr.Row():\n")
        new_lines.append("                        fetch_weather_btn = gr.Button(\"🌤️ Fetch Current Weather\", size=\"sm\")\n")
        new_lines.append("                    \n")
        new_lines.append("                    weather_status = gr.Markdown(\"\")\n")
        new_lines.append("                    \n")
        new_lines.append("                    current_temp = gr.Slider(\n")
        new_lines.append("                        -20, 40, step=0.1, label=\"Current Temperature (°C)\", value=15.0\n")
        new_lines.append("                    )\n")
        new_lines.append("                    \n")
        
        # Add predict button
        new_lines.append(lines[i])
        i += 1
        continue
    
    # 4. Update button connections at the end
    if "# Connect button to prediction function" in line:
        new_lines.append("    # Connect weather fetch button\n")
        new_lines.append("    fetch_weather_btn.click(\n")
        new_lines.append("        fn=fetch_weather_for_country,\n")
        new_lines.append("        inputs=[country_id],\n")
        new_lines.append("        outputs=[current_temp, weather_status],\n")
        new_lines.append("    )\n")
        new_lines.append("    \n")
        new_lines.append("    # Connect prediction button\n")
        i += 1
        continue
    
    # Default: keep the line as is
    new_lines.append(line)
    i += 1

# Write the modified file
with open("App/energy_app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("✅ Successfully updated energy_app.py with weather API integration")
