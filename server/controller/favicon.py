from flask import Blueprint, send_from_directory
from util.favicon import FaviconDto

Favicon = FaviconDto.Favicon

@Favicon.route('/videos/bg_main.mp4')
def video_bg_main():
  return send_from_directory('build/videos', 'bg_main.mp4')

@Favicon.route('/favicon_152.png')
def favicon152():
  return send_from_directory('build', 'favicon_152.png')

@Favicon.route('/favicon_32.png')
def favicon32():
  return send_from_directory('build', 'favicon_32.png')

@Favicon.route('/favicon_16.png')
def favicon16():
  return send_from_directory('build', 'favicon_16.png')

@Favicon.route('/favicon_32.ico')
def favicon_ico32():
  return send_from_directory('build', 'favicon_32.ico')

@Favicon.route('/favicon_16.ico')
def favicon_ico16():
  return send_from_directory('build', 'favicon_16.ico')

@Favicon.route('/manifest.json')
def manifest():
  return send_from_directory('build', 'manifest.json')