import React, { useState } from "react";
import {
  SafeAreaView, ScrollView, View, Text, Pressable, StyleSheet,
  ActivityIndicator, Image
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";

const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function App() {
  const [photo, setPhoto] = useState(null);
  const [enhanced, setEnhanced] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  async function choosePhoto() {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setMessage("Please allow photo-library access to choose a photo.");
      return;
    }
    const selected = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"], allowsEditing: false, quality: 1,
    });
    if (!selected.canceled && selected.assets?.[0]) {
      setPhoto(selected.assets[0]);
      setEnhanced("");
      setMessage("");
    }
  }

  async function enhancePhoto() {
    if (!photo) return;
    setBusy(true);
    setMessage("");
    try {
      const form = new FormData();
      form.append("file", {
        uri: photo.uri,
        name: photo.fileName || "photo.jpg",
        type: photo.mimeType || "image/jpeg",
      });
      const response = await fetch(API_URL + "/api/enhance", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Photo enhancement failed.");
      setEnhanced("data:image/png;base64," + data.image_base64);
      setMessage(data.note || "Your enhanced photo is ready.");
    } catch (error) {
      setMessage(error.message || "Could not connect to the enhancement API.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" />
      <ScrollView contentContainerStyle={styles.page}>
        <View style={styles.header}>
          <View style={styles.logo}><Text style={styles.logoText}>✳</Text></View>
          <View><Text style={styles.brand}>AcneClear AI</Text><Text style={styles.subbrand}>PHOTO ENHANCEMENT</Text></View>
        </View>
        <View style={styles.hero}>
          <Text style={styles.eyebrow}>PHOTO RESTORATION, MADE SIMPLE</Text>
          <Text style={styles.title}>Enhance your photo.</Text>
          <Text style={styles.body}>Upload a clear face photo to restore facial details and gently soften texture.</Text>
          <Pressable style={styles.button} onPress={choosePhoto}>
            <Text style={styles.buttonText}>{photo ? "Choose another photo" : "Choose a photo"}</Text>
          </Pressable>
        </View>
        {photo && (
          <View style={styles.panel}>
            <Text style={styles.sectionTitle}>Original photo</Text>
            <Image source={{ uri: photo.uri }} style={styles.photo} resizeMode="contain" />
            <Pressable style={[styles.button, busy && styles.disabled]} onPress={enhancePhoto} disabled={busy}>
              {busy ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>Enhance photo</Text>}
            </Pressable>
          </View>
        )}
        {!!enhanced && (
          <View style={styles.panel}>
            <Text style={styles.sectionTitle}>Enhanced photo</Text>
            <Image source={{ uri: enhanced }} style={styles.photo} resizeMode="contain" />
          </View>
        )}
        {!!message && <Text style={styles.message}>{message}</Text>}
        <Text style={styles.disclaimer}>Photos are processed in memory and are not saved by this app. GFPGAN is a general face-restoration model, not an acne-specific removal or medical diagnosis system.</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:{flex:1,backgroundColor:"#F8FAFC"}, page:{padding:20,paddingBottom:40},
  header:{flexDirection:"row",alignItems:"center",gap:11,marginBottom:28},
  logo:{width:40,height:40,borderRadius:12,backgroundColor:"#243B63",alignItems:"center",justifyContent:"center"},
  logoText:{color:"#fff",fontSize:22}, brand:{fontSize:18,fontWeight:"800",color:"#243B63"},
  subbrand:{fontSize:9,letterSpacing:1.5,color:"#8291A4",marginTop:3},
  hero:{backgroundColor:"#EAF1F8",borderRadius:18,padding:22,marginBottom:18},
  eyebrow:{fontSize:9,letterSpacing:1.5,fontWeight:"800",color:"#6F89A8"},
  title:{fontSize:29,lineHeight:36,fontWeight:"800",color:"#243B63",marginTop:13},
  body:{fontSize:13,lineHeight:21,color:"#687990",marginTop:9,marginBottom:17},
  button:{backgroundColor:"#243B63",borderRadius:10,minHeight:46,padding:13,alignItems:"center",justifyContent:"center",marginTop:12},
  disabled:{opacity:0.6},buttonText:{color:"#fff",fontWeight:"700",fontSize:13},
  panel:{backgroundColor:"#fff",borderWidth:1,borderColor:"#E3EAF2",borderRadius:15,padding:16,marginTop:14},
  sectionTitle:{fontSize:16,fontWeight:"800",color:"#243B63",marginBottom:12},
  photo:{width:"100%",height:340,backgroundColor:"#F5F7FA",borderRadius:10},
  message:{color:"#61758D",fontSize:12,lineHeight:18,marginTop:13},
  disclaimer:{color:"#8A96A5",fontSize:10,lineHeight:16,marginTop:22},
});
