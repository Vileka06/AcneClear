import React, { useEffect, useState } from "react";
import { SafeAreaView, ScrollView, View, Text, Pressable, StyleSheet, ActivityIndicator } from "react-native";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";
import { Camera, Sparkles, Utensils, Bell, Languages, ShieldCheck } from "lucide-react-native";

const API_URL = "http://localhost:8000"; // On a physical phone, replace localhost with your computer's LAN IP.

export default function App() {
  const [api, setApi] = useState("Checking connection");
  const [photo, setPhoto] = useState(null);
  const [message, setMessage] = useState("");

  useEffect(() => {
    fetch(`${API_URL}/api/health`)
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then(() => setApi("Backend connected"))
      .catch(() => setApi("Backend not connected"));
  }, []);

  async function choosePhoto() {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setMessage("Please allow photo-library access to choose an image.");
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      allowsEditing: false,
      quality: 1,
    });
    if (!result.canceled) {
      setPhoto(result.assets[0]);
      setMessage("Photo selected. Upload validation will be connected next.");
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.page}>
        <View style={styles.header}>
          <View style={styles.logo}><Sparkles color="#fff" size={19}/></View>
          <View><Text style={styles.brand}>AcneClear AI</Text><Text style={styles.subbrand}>SKIN WELLNESS STUDIO</Text></View>
        </View>
        <View style={styles.hero}>
          <Text style={styles.eyebrow}>YOUR PERSONAL SKIN WELLNESS SPACE</Text>
          <Text style={styles.title}>Understand your skin.{"\n"}Care with confidence.</Text>
          <Text style={styles.body}>Skin analysis, skincare education, balanced meal ideas and photo restoration in one place.</Text>
          <Pressable style={styles.button} onPress={choosePhoto}><Camera color="#fff" size={18}/><Text style={styles.buttonText}>Choose a skin photo</Text></Pressable>
          {photo && <Text style={styles.selected}>Selected: {photo.fileName || "Your photo"}</Text>}
          {!!message && <Text style={styles.message}>{message}</Text>}
        </View>
        <Text style={styles.sectionTitle}>Your wellness toolkit</Text>
        <Module icon={Camera} title="Skin analysis" detail="Region-based facial skin assessment."/>
        <Module icon={Sparkles} title="Beauty guidance" detail="Morning and evening routine education."/>
        <Module icon={Utensils} title="Diet planner" detail="Balanced meal ideas and food visuals."/>
        <Module icon={ShieldCheck} title="Photo restoration" detail="GFPGAN integration, with clear limitations."/>
        <View style={styles.info}><Languages color="#6888a9"/><Text style={styles.infoText}>English · Kannada · Hindi · Telugu planned</Text></View>
        <View style={styles.info}><Bell color="#6888a9"/><Text style={styles.infoText}>Optional reminders planned</Text></View>
        <Text style={styles.status}>{api}</Text>
        <Text style={styles.disclaimer}>Educational wellness support only. AcneClear AI does not diagnose skin disease or replace a dermatologist.</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

function Module({ icon: Icon, title, detail }) {
  return <View style={styles.module}><View style={styles.moduleIcon}><Icon color="#54799f" size={20}/></View><View style={{flex:1}}><Text style={styles.moduleTitle}>{title}</Text><Text style={styles.moduleText}>{detail}</Text></View></View>;
}

const styles = StyleSheet.create({
  safe:{flex:1,backgroundColor:"#fbfcfe"},page:{padding:22,paddingBottom:40},
  header:{flexDirection:"row",alignItems:"center",gap:11,marginBottom:35},
  logo:{width:39,height:39,borderRadius:12,backgroundColor:"#243b63",alignItems:"center",justifyContent:"center"},
  brand:{fontSize:18,fontWeight:"800",color:"#243b63"},subbrand:{fontSize:8,letterSpacing:1.5,color:"#8998aa",marginTop:3},
  hero:{backgroundColor:"#eaf1f8",borderRadius:22,padding:23,marginBottom:30},
  eyebrow:{fontSize:9,letterSpacing:1.5,fontWeight:"800",color:"#6684a5"},
  title:{fontSize:34,lineHeight:41,fontWeight:"800",color:"#243b63",marginTop:15},
  body:{fontSize:14,lineHeight:22,color:"#6d7b8e",marginTop:13,marginBottom:20},
  button:{backgroundColor:"#243b63",borderRadius:10,padding:14,flexDirection:"row",alignItems:"center",justifyContent:"center",gap:9},
  buttonText:{color:"#fff",fontWeight:"700",fontSize:13},selected:{color:"#456a8f",fontSize:12,marginTop:12},message:{color:"#61758d",fontSize:12,marginTop:8},
  sectionTitle:{fontSize:20,fontWeight:"800",color:"#243b63",marginBottom:14},
  module:{backgroundColor:"#fff",borderWidth:1,borderColor:"#e4eaf1",borderRadius:13,padding:15,marginBottom:10,flexDirection:"row",alignItems:"center",gap:13},
  moduleIcon:{width:39,height:39,borderRadius:11,backgroundColor:"#eaf1f8",alignItems:"center",justifyContent:"center"},
  moduleTitle:{fontSize:14,fontWeight:"700",color:"#243b63"},moduleText:{fontSize:11,color:"#7a8797",marginTop:4},
  info:{flexDirection:"row",alignItems:"center",gap:10,marginTop:17},infoText:{fontSize:12,color:"#718198"},
  status:{fontSize:11,color:"#6383a3",marginTop:24},disclaimer:{fontSize:10,lineHeight:16,color:"#8995a4",marginTop:10}
});
